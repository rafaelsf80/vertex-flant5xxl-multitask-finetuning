import argparse
import numpy as np
import glob
import json
import logging
import os

from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from transformers import set_seed

from datasets import load_from_disk
import torch
import evaluate
import nltk

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from google.cloud import storage
from google.cloud import aiplatform

MODEL_ID = "google/flan-t5-xxl"   # Model id to use for training, in Hugging face
TRAIN_DATASET_PATH = "data/train" # Path to processed dataset 
TEST_DATASET_PATH  = "data/eval"  # Path to processed dataset
EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8 # Batch size to use for training
PER_DEVICE_EVAL_BATCH_SIZE = 8  # Batch size to use for testing
GENERATION_MAX_LENTH = 129      # Maximum length to use for generation
GENERATION_NUM_BEAMS = 4        # Number of beams to use for generation
LR = 1e-4 # Learning rate to use for training
SEED = 42 # Seed use for training
DEEPSPEED_FILE = "configs/ds_flan_t5_z3_config_bf16.json"             # Path to deepspeed config file
GRADIENT_CHECKPOINTING = True                                         # Whether to use gradient checkpointing
BF_16 = True if torch.cuda.get_device_capability()[0] == 8 else False # Whether to use bf16

# avoid noisy neighbour #3514 https://github.com/deepset-ai/haystack/issues/3514
try:
    nltk.data.find("tokenizers/punkt")
except (OSError, LookupError):
    try:
        nltk.download("punkt")
    except FileExistsError:
        pass

output_directory = os.environ['AIP_MODEL_DIR']

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

# set seed
set_seed(SEED)

# load dataset from disk and tokenizer
train_dataset = load_from_disk(TRAIN_DATASET_PATH)
eval_dataset = load_from_disk(TEST_DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    use_cache=False if GRADIENT_CHECKPOINTING else True  # this is needed for gradient checkpointing
)

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)

# Define compute metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# Define training args
output_dir = "flant5-xxl-checkpoints"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENTH,
    generation_num_beams=GENERATION_NUM_BEAMS,
    fp16=False,  # T5 overflows with fp16
    bf16=BF_16,  # Use BF16 if available
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    deepspeed=DEEPSPEED_FILE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    # tensorboard
    logging_dir=os.environ["AIP_TENSORBOARD_LOG_DIR"], 
    report_to=["tensorboard"],  
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,  # avoiding OOM errors

    # push to hub parameters
    push_to_hub=False, # no puhsing to hub
    hub_strategy="every_save",
    hub_model_id=None,
    hub_token=None 
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start training and evaluation
logging.info("Training ....")
trainer.train()
logging.info("Evaluating ....")
trainer.evaluate()


# Save tokenizer, metrics and model locally
logging.info("Saving model and tokenizer locally ....")
tokenizer.save_pretrained(f'model_tokenizer')
trainer.save_model(f'model_output')
logging.info('Saving metrics...')
with open(os.path.join(f'model_output', 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

output_directory = os.environ['AIP_MODEL_DIR']
logging.info("Saving model and tokenizer to GCS ....")
logging.info(f'Exporting SavedModel to: {output_directory}')

# extract GCS bucket_name from AIP_MODEL_DIR, ex: argolis-vertex-europewest4
bucket_name = output_directory.split("/")[2] # without gs://

# extract GCS object_name from AIP_MODEL_DIR, ex: aiplatform-custom-training-2023-02-22-16:31:12.167/model/
object_name = "/".join(output_directory.split("/")[3:])

directory_path = "model_output" # local
# Upload model to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


directory_path = "model_tokenizer" # local
# Upload tokenizer to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

# TODO: Upload metrics to Vertex AI Experiments


