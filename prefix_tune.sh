# bart
# model_name=bart and model_checkpoint=Salesforce/bart-large-xsum-samsum
# T-5
# model_name=t5 and model_checkpoint=t5-large-samsum

# bart, 1% dataset, Multi Domain
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ds2/scripts/train_ds2.py \
#     --dev_batch_size=8 \
#     --test_batch_size=8 \
#     --train_batch_size=2 \
#     --n_epochs=100 \
#     --num_beams=1 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.01 \
#     --grad_acc_steps=1 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --mode=finetune \
#     --exp_name=bart-MD-1 \
#     --seed=577 \
#     --version=2.1 \
#     --GPU=4

# From Salesforce/bart-large-xsum-samsum, Only prefixtune, not testing
# bart, 10% dataset, Multi Domain
CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2.py \
    --dev_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.1 \
    --gradient_accumulation_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --model_name_or_path=Salesforce/bart-large-xsum-samsum \
    --mode=prefixtune \
    --exp_name=bart-MD-prefixtune-1 \
    --seed=577 \
    --version=2.1 \
    --GPU=1 \
    --pf_mode=multiwoz \
    --tuning_mode=prefixtune \
    --optim_prefix=yes \
    --preseqlen=200 \
    --prefix_mode=activation \
    --warmup_steps=100 \
    --max_steps=400 \
    --eval_steps=50

# # From Salesforce/bart-large-xsum-samsum, Only prefixtune, not testing
# # bart, 10% dataset, Multi Domain
# CUDA_VISIBLE_DEVICES=0,1 python ds2/pftune_scripts/pftune_ds2.py \
#     --dev_batch_size=8 \
#     --train_batch_size=2 \
#     --n_epochs=100 \
#     --num_beams=1 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.1 \
#     --gradient_accumulation_steps=1 \
#     --model_name=t5 \
#     --model_checkpoint=t5-large-samsum \
#     --model_name_or_path=t5-large-samsum \
#     --mode=prefixtune \
#     --exp_name=bart-MD-prefixtune-1 \
#     --seed=577 \
#     --version=2.1 \
#     --GPU=2 \
#     --pf_mode=multiwoz \
#     --tuning_mode=prefixtune \
#     --optim_prefix=yes \
#     --preseqlen=200 \
#     --prefix_mode=activation \
#     --warmup_steps=100 \
#     --max_steps=400 \
#     --eval_steps=50
    