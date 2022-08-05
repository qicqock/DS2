import itertools
import json
import time

import pytorch_lightning as pl
import nltk; nltk.download('punkt')
import numpy as np
import rouge

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AdamW

from ds2.utils.state_sum_converter import get_converter
from ds2.utils.evaluate import get_acc
from ds2.utils.evaluate import get_template_acc
from ds2.utils.evaluate import EXPERIMENT_DOMAINS


# inherit the pytorch-lightning module
class DS2(pl.LightningModule):
    def __init__(self, args, tokenizer, sum_model, qa_model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.sum_model = sum_model
        if self.args["use_qa_deconverter"]:
            self.qa_model = qa_model # None
        self.lr = args["lr"]
        self.blank = "____"

        # converter
        self.converter = get_converter(args['state_converter'])
        # evaluator
        self.evaluator = rouge.Rouge(
            metrics=['rouge-n'],
            max_n=4,
            limit_length=True,
            length_limit=100,
            length_limit_type='words',
            apply_avg=False,
            apply_best=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True
        )

    # the complete training loop
    def training_step(self, batch, batch_idx):
        self.sum_model.train() # sum_model = AutoModelForSeq2SeqLM.from_pretrained(args["model_checkpoint"])

        outputs = self.sum_model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        )

        return {'loss': outputs.loss, 'log': {'train_loss': outputs.loss.detach()}}

    # evaluate for each step
    def eval_step(self, batch, batch_idx):
        self.sum_model.eval()

        outputs = self.sum_model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        )

        return outputs.loss.item()

    # predict the result for each step
    def pred_step(self, batch, batch_idx):
        self.sum_model.eval()
        # In HuggingFace, "model.generate()" means decoding operation of model.
        pred_summary_token = self.sum_model.generate(
            batch["encoder_input"], # select initial input for generation
            num_beams=self.args["num_beams"], # beam search
            min_length=5,
            max_length=100,
            early_stopping=True,
        )

        return {
            "pred_summary_token": pred_summary_token,
            "gold_state": batch["slot_values"],
            "gold_summary": batch["output_text"],
            "eval_slots": batch["eval_slots"],
        }

    # if we need to do something with all the outputs of each xxx_step(), override the "xxx_epoch_end()" method
    def eval_epoch_end(self, outputs):
        res = {}
        res["loss"] = np.mean(outputs)
        print(res)
        return res

    # predict the result for each epoch 
    # Make gold template from y:states and compare with predicted summarys
    def pred_epoch_end(self, outputs, mode="val"):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}

        # pred_summary : tokenized predicted summary
        pred_summary = [
            self.tokenizer.decode(_sum, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _sum in outputs["pred_summary_token"]
        ]

        # qa_model = None
        if self.args["use_qa_deconverter"]:
            pred_state = self.qa_model.sum_to_state(pred_summary, outputs["eval_slots"])
        else:
            pred_state = [self.converter.sum_to_state(_sum) for _sum in pred_summary]

        res = get_acc(pred_state, outputs["gold_state"], outputs["eval_slots"])

        # converts y:state into summary to make gold template
        gold_templates = [
            # state -> summary
            self.converter.state_to_sum(_ds, is_for_template=True, blank=self.blank)
            for _ds in outputs["gold_state"]
        ]

        # compare pred_summary and gold template
        template_acc = get_template_acc(pred_summary, gold_templates, self.blank)
        rouge_score = self.evaluator.get_scores(pred_summary, outputs["gold_summary"])["rouge-4"]["f"]
        bleu_score = [
            sentence_bleu(
                [ref.split()],
                hyp.split(),
                smoothing_function=SmoothingFunction().method1
            )
            for ref, hyp in zip(outputs["gold_summary"], pred_summary)
        ]
        res.update({
            'rouge': rouge_score,
            'bleu': np.mean(bleu_score),
            'template_acc': template_acc,
        })

        # save the sample
        samples = {"gold_summary": outputs["gold_summary"], "gold_state": outputs["gold_state"], "pred_summary": pred_summary, "pred_state": pred_state}
        self.save_samples(samples, f'{str(res["jga"])}_{mode}')

        print(res)

        return res

    """ validation """
    # the complete validation loop
    def validation_step(self, batch, batch_idx):
        if self.args["eval_loss_only"]: 
            return self.eval_step(batch, batch_idx)
        else:
            return self.pred_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if self.args["eval_loss_only"]:
            res = {f'val_{k}': v for k, v in self.eval_epoch_end(outputs).items()}
        else:
            res = {f'val_{k}': v for k, v in self.pred_epoch_end(outputs, "val").items()}
        self.log_dict(res)
        return res

    """ test """
    # the complete test step
    def test_step(self, batch, batch_idx):
        return self.pred_step(batch, batch_idx) # sum_model.generate()

    def test_epoch_end(self, outputs):
        res = {f'test_{k}': v for k, v in self.pred_epoch_end(outputs, "test").items()}
        self.log_dict(res)
        return res

    # optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def save_samples(self, samples, name):
        if self.args["save_samples"] > 0:
            output_fields = ['only_domain', 'fewshot', 'grad_acc_steps', 'train_batch_size', 'state_converter']
            output_name = '_'.join([str(self.args[k]) for k in output_fields]) + '_' + name + '_' + str(round(time.time()))
            filename = f'./samples_data/{output_name}.json'
            with open(filename, 'w') as f:
                json.dump({k: v[:self.args['save_samples']] for k, v in samples.items()}, f)