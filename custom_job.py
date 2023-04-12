""" Custom training pipeline """

from google.cloud import aiplatform
from datetime import datetime

BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE = 'projects/989788194604/locations/europe-west4/tensorboards/9215314815646433280'
TRAIN_IMAGE="europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_flan_t5_xxl:latest"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")


# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)

job = aiplatform.CustomContainerTrainingJob(
    display_name="flant5xxl_deepspeed_" + TIMESTAMP,
    container_uri=TRAIN_IMAGE,
    #command=["deepspeed", "--num_gpus=8", "run_seq2seq_deepspeed.py"],
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest"
)

model = job.run(
    model_display_name='flan-t5-xxl-finetuning-gpu-deepspeed',
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE,
    boot_disk_size_gb=600,
    machine_type="a2-highgpu-8g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count = 8,
)
