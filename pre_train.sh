# bart
# model_name=bart and model_checkpoint=Salesforce/bart-large-xsum-samsum
# T-5
# model_name=t5 and model_checkpoint=jaynlp/t5-large-samsum

# MultiWoZ 2.0
CUDA_VISIBLE_DEVICES=0,1,2,3 python ds2/scripts/train_ds2.py \
    --dev_batch_size=8 \
    --test_batch_size=8 \
    --train_batch_size=2 \
    --n_epochs=100 \
    --num_beams=1 \
    --test_num_beams=1 \
    --val_check_interval=1.0 \
    --fewshot=0.01 \
    --grad_acc_steps=1 \
    --model_name=bart \
    --model_checkpoint=Salesforce/bart-large-xsum-samsum \
    --except_domain=attraction \
    --mode=finetune \
    --exp_name=bart-CD-1-Attr-pre \
    --seed=577 \
    --version=2.0 \
    --GPU=4 # set the number of gpus