# bart: model_name=bart and model_checkpoint=Salesforce/bart-large-xsum-samsum
# T-5: model_name=t5 and model_checkpoint=t5-large-samsum
# Training batch size was 2 for T5-large, and 1 for BART.
# In DS2 paper, they set accumulate grad batches options to 1, 5, 10, 100 for 1%, 5%, 10%, 100% few shot learning.
# In PfTune paper, mid_dim is 800 for summarization task.

# Only prefixtuning for Multi Domain(MD), not testing

# bart, 1% dataset
# CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2.py \
#     --dev_batch_size=64 \
#     --train_batch_size=1 \
#     --n_epochs=20 \
#     --num_beams=1 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.01 \
#     --gradient_accumulation_steps=1 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --model_name_or_path=Salesforce/bart-large-xsum-samsum \
#     --mode=prefixtune \
#     --exp_name=bart-MD-prefixtune-1% \
#     --seed=47 \
#     --version=2.1 \
#     --GPU=1 \
#     --pf_mode=multiwoz \
#     --tuning_mode=prefixtune \
#     --optim_prefix=yes \
#     --preseqlen=200 \
#     --prefix_mode=activation \
#     --warmup_steps=100 \
#     --max_steps=400 \
#     --eval_steps=50

# bart, 5% dataset, Multi Domain
# CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2.py \
#     --dev_batch_size=64 \
#     --train_batch_size=1 \
#     --n_epochs=30 \
#     --num_beams=1 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.05 \
#     --gradient_accumulation_steps=5 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --model_name_or_path=Salesforce/bart-large-xsum-samsum \
#     --mode=prefixtune \
#     --exp_name=bart-MD-prefixtune-5%-preseqlen=10-mid_dim=256 \
#     --seed=11 \
#     --version=2.1 \
#     --GPU=1 \
#     --pf_mode=multiwoz \
#     --tuning_mode=prefixtune \
#     --optim_prefix=yes \
#     --preseqlen=10 \
#     --prefix_mode=activation \
#     --warmup_steps=100 \
#     --max_steps=400 \
#     --eval_steps=50 \
#     --mid_dim=256

# # bart, 10% dataset, Multi Domain
# CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2.py \
#     --dev_batch_size=64 \
#     --train_batch_size=1 \
#     --n_epochs=30 \
#     --num_beams=1 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.1 \
#     --gradient_accumulation_steps=10 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --model_name_or_path=Salesforce/bart-large-xsum-samsum \
#     --mode=prefixtune \
#     --exp_name=bart-MD-prefixtune-10% \
#     --seed=47 \
#     --version=2.1 \
#     --GPU=1 \
#     --pf_mode=multiwoz \
#     --tuning_mode=prefixtune \
#     --optim_prefix=yes \
#     --preseqlen=200 \
#     --prefix_mode=activation \
#     --warmup_steps=100 \
#     --max_steps=400 \
#     --eval_steps=50


# bart, Multi Domain, preseqlen=10, mid_dim=800

# 10% dataset
CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2.py \
    --dev_batch_size=64 \
    --train_batch_size=16 \
    --n_epochs=30 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.1 \
    --gradient_accumulation_steps=10 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --model_name_or_path=Salesforce/bart-large-xsum-samsum \
    --mode=prefixtune \
    --exp_name=bart-MD-prefixtune-10%-preseqlen=10-mid_dim=800 \
    --seed=11 \
    --version=2.1 \
    --GPU=1 \
    --pf_mode=multiwoz \
    --tuning_mode=prefixtune \
    --optim_prefix=yes \
    --preseqlen=10 \
    --prefix_mode=activation \
    --warmup_steps=100 \
    --max_steps=400 \
    --eval_steps=50 \
    --mid_dim=800 \
    --patience=10

# # 100% dataset
# CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2.py \
#     --dev_batch_size=64 \
#     --train_batch_size=16 \
#     --n_epochs=30 \
#     --num_beams=1 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=1 \
#     --gradient_accumulation_steps=10 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --model_name_or_path=Salesforce/bart-large-xsum-samsum \
#     --mode=prefixtune \
#     --exp_name=bart-MD-prefixtune-100%-preseqlen=10-mid_dim=800 \
#     --seed=47 \
#     --version=2.1 \
#     --GPU=1 \
#     --pf_mode=multiwoz \
#     --tuning_mode=prefixtune \
#     --optim_prefix=yes \
#     --preseqlen=10 \
#     --prefix_mode=activation \
#     --warmup_steps=100 \
#     --max_steps=400 \
#     --eval_steps=50 \
#     --mid_dim=800 \
#     --patience=10