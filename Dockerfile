FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest

WORKDIR /

LABEL com.nvidia.volumes.needed=nvidia_driver

# env variables for proper GPU setup
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# copy deepspeed config file
COPY configs /configs

# copy preprocessed data. Need to run "create_flan_t5_cnn_dataset.py" first
COPY data /data

# copy deepspeed launcher
COPY run_seq2seq_deepspeed.py run_seq2seq_deepspeed.py

# install dependencies
RUN pip3 --timeout=300 --no-cache-dir install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Sets up the entry point to invoke the trainer with deepspeed
ENTRYPOINT ["deepspeed", "--num_gpus=8", "run_seq2seq_deepspeed.py"]


