# bart
# model_name=bart and model_checkpoint=Salesforce/bart-large-xsum-samsum
# T-5
# model_name=t5 and model_checkpoint=jaynlp/t5-large-samsum

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

# bart, 10% dataset, Multi Domain
CUDA_VISIBLE_DEVICES=0,1,2,3 python ds2/scripts/train_ds2.py \
    --dev_batch_size=8 \
    --test_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.1 \
    --grad_acc_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --mode=finetune \
    --exp_name=bart-MD-10 \
    --seed=577 \
    --version=2.1 \
    --GPU=4