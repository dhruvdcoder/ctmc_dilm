# Setup

Create a new environment using conda.

```bash
git clone --recurse-submodules https://github.com/dhruvdcoder/ctmc_dilm.git
conda create -p .venv_idlm python=3.11.10 pip ipykernel -y
conda activate ./.venv_idlm
pip install -e  lib/xlm-core
```

Create `.env` file in the root directory from where you plan to run the  experiments, edit and add the following environment variables in the file:
```bash
# wandb
WANDB_ENTITY=???
WANDB_PROJECT=???
# paths to appropriate directories
DATA_DIR=data
HF_HOME=hf_home
HF_DATASETS_CACHE=hf_datasets_cache
# output
LOG_DIR=logs
# misc
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
# hydra
HYDRA_FULL_ERROR=1
OC_CAUSE=1
# emails from slurm scheduler if applicable
EMAIL=???
# torch compile logs
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
```

# Download pretrained weights

```bash
pip install gdown
gdown --folder -O logs <gdrive_id>
```

# Eval

**Evaluating from a full training checkpoint**

```bash
DATASET=owt # owt, lm1b
CKPT_PATH=<logs_dir>/${DATASET}_idlm/checkpoints/26-300000.ckpt
top_p=1.0
max_steps=1024
second_top=1

xlm job_name=${DATASET}_idlm_eval_${top_p}_${max_steps}_${second_top} job_type=eval experiment=[${DATASET}_idlm,gpt2_generative_perplexity] \
++eval.checkpoint_path=${CKPT_PATH} \
per_device_batch_size=10 \
per_device_val_batch_size=10 \
global_batch_size=10 \
trainer_strategy=single_device \
++trainer.precision=32-true \
compile=false \
datamodule.dataset_managers.val.unconditional_prediction.num_examples=1000 \
~datamodule.dataset_managers.val.lm \
++predictor.max_steps=${max_steps} \
++predictor.p=${top_p} \
++predictor.second_top=1
```

# Train DILM-S

## LM1B

```bash
# prepare the data
xlm job_name=lm1b_idlm_prepare_data job_type=prepare_data experiment=lm1b_idlm
# start training
xlm job_name=lm1b_idlm job_type=train experiment=lm1b_idlm per_device_batch_size=64 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=true
```

## OpenWebText (1024 sequence length split)
```bash
# prepare the data
xlm job_name=owt_idlm_prepare_data job_type=prepare_data experiment=owt_idlm
# start training
xlm job_name=owt_idlm job_type=train experiment=owt_idlm per_device_batch_size=32 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=true loggers=wandb
```

# Acknowledgments
The code uses [xlm-core](https://github.com/dhruvdcoder/xlm-core) as the rapid experiment framework.
The code for data pipeline for the graph traversal experiments is from [ILM](https://github.com/dhruvdcoder/ILM) 

# Cite
```
@inproceedings{
patel2026a,
title={A Continuous Time Markov Chain Framework for Insertion Language Models},
author={Dhruvesh Patel and Benjamin Rozonoyer and Soumitra Das and Tahira Naseem and Tim G. J. Rudner and Andrew McCallum},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=nCyV21FmUI}
}
```
