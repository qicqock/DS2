# bart: model_name=bart and model_checkpoint=Salesforce/bart-large-xsum-samsum
# T-5: model_name=t5 and model_checkpoint=t5-large-samsum
# Training batch size was 2 for T5-large, and 1 for BART.
# we set accumulate grad batches options to 1, 5, 10, 100 for 1%, 5%,10%, 100% few shot learning.

# PrefixTuning testing

# # bart, 1% dataset, Multi Domain
CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2_test.py \
    --dev_batch_size=64 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.01 \
    --gradient_accumulation_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --model_name_or_path=Salesforce/bart-large-xsum-samsum \
    --prefixModel_name_or_path=ds2/logs/bart-MD-prefixtune-1%/prefixtune/47/checkpoint-epoch=8-val_jga=0.2581/pytorch_model.bin \
    --mode=prefixtune \
    --exp_name=bart-MD-prefixtune-1% \
    --seed=47 \
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

# bart, 5% dataset, Multi Domain
# CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2_test.py \
#     --dev_batch_size=64 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.05 \
#     --gradient_accumulation_steps=5 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --model_name_or_path=Salesforce/bart-large-xsum-samsum \
#     --prefixModel_name_or_path=ds2/logs/bart-MD-prefixtune-5%/prefixtune/47/checkpoint-epoch=6-val_jga=0.3961/pytorch_model.bin \
#     --mode=prefixtune \
#     --exp_name=bart-MD-prefixtune-5% \
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

# # bart, 10% dataset, Multi Domain
# CUDA_VISIBLE_DEVICES=0 python ds2/pftune_scripts/pftune_ds2_test.py \
#     --dev_batch_size=64 \
#     --test_num_beams=1 \
#     --val_check_interval=1.0 \
#     --fewshot=0.1 \
#     --gradient_accumulation_steps=10 \
#     --model_name=bart \
#     --model_checkpoint=Salesforce/bart-large-xsum-samsum \
#     --model_name_or_path=Salesforce/bart-large-xsum-samsum \
#     --prefixModel_name_or_path=ds2/logs/bart-MD-prefixtune-10%/prefixtune/47/checkpoint-epoch=5-val_jga=0.4252/pytorch_model.bin \
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