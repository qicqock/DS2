# ds2 dependent
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

# prefixtuning dependent
import argparse
import glob
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers import MBartTokenizer, T5ForConditionalGeneration

from transformers.modeling_bart import shift_tokens_right # For transformers version 3.2.0
# from transformers.models.bart.modeling_bart import shift_tokens_right # For transformers version 4.11.3 

from transformers.modeling_outputs import Seq2SeqLMOutput

from ds2.pftune_helper.utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    use_task_specific_params,
)
from ds2.pftune_models.lightning_base import PrefixTransformer, add_generic_args, generic_train

logger = logging.getLogger(__name__)

# prefix Module
# Manage gpu, data, evaluation, freezing, and forward
class PrefixSummarizationModule(PrefixTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1: # default = None
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None: # default = None
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")
        
        # default: self.mode = summarization
        # self.model means PrefixTuningModel
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")
        # save_git_info(self.hparams.output_dir)
        # self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        # self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type # e.g.) bart
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )

        # define split. default: -1(use_alls)
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        # define target maximum length
        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        # if self.hparams.freeze_embeds:
        #     self.freeze_embeds()

        # In init, freeze all parameters of the model
        freeze_params(self.seq2seq_model)
        assert_all_frozen(self.seq2seq_model)
        print('FREEZING ENTIRE seq2seq model.')
        # if self.hparams.freeze_encoder:
        #     freeze_params(self.model.get_encoder())
        #     assert_all_frozen(self.model.get_encoder())


        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers

        # set decoder_start_token_id
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id

        self.dataset_class = (
            Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )

        # beam search parameters
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}. Need an integer > 1"

        # eval_max_gen_length
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        self.training_acc_across_batches_at_curr_epoch = []

        self.eval_max_length = 62
        self.eval_min_length = 11
        self.eval_beams =6
        print('for decoding, eval_max_length={}, '
              'eval_min_length={}, eval_beams={}'.format(self.eval_max_length, self.eval_min_length, self.eval_beams))

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        if self.model_type == "t5":
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)
        elif self.model_type == "fsmt":
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        else: # e.g.) bart
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, gpt2_model=self.seq2seq_model, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    # calculate loss between batch["input_ids"] and batch["labels"]
    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id

        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)

        # outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
        #                use_prefix=True, return_dict=True, labels=tgt_ids)
        #
        # return (outputs.loss,)

        # self() calls __call__()
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       use_prefix=True)

        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0: # default: 0.0
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            # print(lm_logits.shape, tgt_ids.shape, lm_logits.shape[-1] )
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()

        # print('hi', loss_tensors[0].item())
        self.training_acc_across_batches_at_curr_epoch.append(loss_tensors[0].item())
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs}

    # mean accuracy of loss across batches at current batches
    def on_epoch_end(self):
        train_acc_mean = np.mean(self.training_acc_across_batches_at_curr_epoch)
        print('train_loss = {}'.format(train_acc_mean))
        # print('train_PPL = {}'.format(train_acc_mean.exp()))
        self.training_acc_across_batches_per_epoch = []  # reset for next epoch

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        print(loss)
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)
        # return calculate_bleu(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        # TODO(LISA)
        # write the prompt generation from self.model.
        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        # get the prompt:
        bsz = batch["input_ids"].size(0)
        prefix_prompt = self.model.get_prompt(bsz=bsz, sample_size=self.eval_beams)
        
        generated_ids = self.seq2seq_model.generate(
            batch["input_ids"],
            past_key_values = prefix_prompt,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            length_penalty=self.hparams.length_penalty,
            use_prefix=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            min_length=self.eval_min_length,
            max_length=self.eval_max_length,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # print('INPUT:', self.ids_to_clean_text(batch["input_ids"]))
        # print(preds, target)
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    # dataloader
    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.dev_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.dev_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        PrefixTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=512, #1024
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56, #56
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  #142 # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142, #142
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task_mode", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser


# class SummarizationModule(BaseTransformer):
#     mode = "summarization"
#     loss_names = ["loss"]
#     metric_names = ROUGE_KEYS
#     default_val_metric = "rouge2"

#     def __init__(self, hparams, **kwargs):
#         if hparams.sortish_sampler and hparams.gpus > 1:
#             hparams.replace_sampler_ddp = False
#         elif hparams.max_tokens_per_batch is not None:
#             if hparams.gpus > 1:
#                 raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
#             if hparams.sortish_sampler:
#                 raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")
#         super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
#         use_task_specific_params(self.model, "summarization")
#         save_git_info(self.hparams.output_dir)
#         self.metrics_save_path = Path(self.output_dir) / "metrics.json"
#         self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
#         pickle_save(self.hparams, self.hparams_save_path)
#         self.step_count = 0
#         self.metrics = defaultdict(list)
#         self.model_type = self.config.model_type
#         self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

#         self.dataset_kwargs: dict = dict(
#             data_dir=self.hparams.data_dir,
#             max_source_length=self.hparams.max_source_length,
#             prefix=self.model.config.prefix or "",
#         )
#         n_observations_per_split = {
#             "train": self.hparams.n_train,
#             "val": self.hparams.n_val,
#             "test": self.hparams.n_test,
#         }
#         self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

#         self.target_lens = {
#             "train": self.hparams.max_target_length,
#             "val": self.hparams.val_max_target_length,
#             "test": self.hparams.test_max_target_length,
#         }
#         assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
#         assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
#         if self.hparams.freeze_embeds:
#             self.freeze_embeds()
#         else:
#             print('THE EMBEDDING IS NOT FROZEN.')
#         if self.hparams.freeze_encoder:
#             freeze_params(self.model.get_encoder())
#             assert_all_frozen(self.model.get_encoder())
#         else:
#             print('THE ENCODER IS NOT FROZEN.')

#         self.hparams.git_sha = get_git_info()["repo_sha"]
#         self.num_workers = hparams.num_workers
#         self.decoder_start_token_id = None  # default to config
#         if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
#             self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
#             self.model.config.decoder_start_token_id = self.decoder_start_token_id
#         self.dataset_class = (
#             Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
#         )
#         self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
#         assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}. Need an integer > 1"
#         if self.hparams.eval_max_gen_length is not None:
#             self.eval_max_length = self.hparams.eval_max_gen_length
#         else:
#             self.eval_max_length = self.model.config.max_length
#         self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

#     def freeze_embeds(self):
#         """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
#         if self.model_type == "t5":
#             freeze_params(self.model.shared)
#             for d in [self.model.encoder, self.model.decoder]:
#                 freeze_params(d.embed_tokens)
#         elif self.model_type == "fsmt":
#             for d in [self.model.model.encoder, self.model.model.decoder]:
#                 freeze_params(d.embed_positions)
#                 freeze_params(d.embed_tokens)
#         else:
#             freeze_params(self.model.model.shared)
#             for d in [self.model.model.encoder, self.model.model.decoder]:
#                 freeze_params(d.embed_positions)
#                 freeze_params(d.embed_tokens)

#     def forward(self, input_ids, **kwargs):
#         return self.model(input_ids, **kwargs)

#     def ids_to_clean_text(self, generated_ids: List[int]):
#         gen_text = self.tokenizer.batch_decode(
#             generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#         )
#         return lmap(str.strip, gen_text)

#     def _step(self, batch: dict) -> Tuple:
#         pad_token_id = self.tokenizer.pad_token_id
#         src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
#         tgt_ids = batch["labels"]
#         if isinstance(self.model, T5ForConditionalGeneration):
#             decoder_input_ids = self.model._shift_right(tgt_ids)
#         else:
#             decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)

#         outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
#         lm_logits = outputs[0]
#         if self.hparams.label_smoothing == 0:
#             # Same behavior as modeling_bart.py, besides ignoring pad_token_id
#             ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

#             assert lm_logits.shape[-1] == self.vocab_size
#             loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
#         else:
#             lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
#             loss, nll_loss = label_smoothed_nll_loss(
#                 lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
#             )
#         return (loss,)

#     @property
#     def pad(self) -> int:
#         return self.tokenizer.pad_token_id

#     def training_step(self, batch, batch_idx) -> Dict:
#         loss_tensors = self._step(batch)

#         logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
#         # tokens per batch
#         logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
#         logs["bs"] = batch["input_ids"].shape[0]
#         logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
#         logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
#         # TODO(SS): make a wandb summary metric for this
#         return {"loss": loss_tensors[0], "log": logs}

#     def validation_step(self, batch, batch_idx) -> Dict:
#         return self._generative_step(batch)

#     def validation_epoch_end(self, outputs, prefix="val") -> Dict:
#         self.step_count += 1
#         losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
#         loss = losses["loss"]
#         print(loss)
#         generative_metrics = {
#             k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
#         }
#         metric_val = (
#             generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
#         )
#         metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
#         generative_metrics.update({k: v.item() for k, v in losses.items()})
#         losses.update(generative_metrics)
#         all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
#         all_metrics["step_count"] = self.step_count
#         self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
#         preds = flatten_list([x["preds"] for x in outputs])
#         return {
#             "log": all_metrics,
#             "preds": preds,
#             f"{prefix}_loss": loss,
#             f"{prefix}_{self.val_metric}": metric_tensor,
#         }

#     def calc_generative_metrics(self, preds, target) -> Dict:
#         return calculate_rouge(preds, target)

#     def _generative_step(self, batch: dict) -> dict:
#         t0 = time.time()

#         # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
#         generated_ids = self.model.generate(
#             batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             use_cache=True,
#             length_penalty=self.hparams.length_penalty,
#             decoder_start_token_id=self.decoder_start_token_id,
#             num_beams=self.eval_beams,
#             max_length=self.eval_max_length,
#         )
#         gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
#         preds: List[str] = self.ids_to_clean_text(generated_ids)
#         target: List[str] = self.ids_to_clean_text(batch["labels"])
#         loss_tensors = self._step(batch)
#         base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
#         rouge: Dict = self.calc_generative_metrics(preds, target)
#         summ_len = np.mean(lmap(len, generated_ids))
#         base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
#         return base_metrics

#     def test_step(self, batch, batch_idx):
#         return self._generative_step(batch)

#     def test_epoch_end(self, outputs):
#         return self.validation_epoch_end(outputs, prefix="test")

#     def get_dataset(self, type_path) -> Seq2SeqDataset:
#         n_obs = self.n_obs[type_path]
#         max_target_length = self.target_lens[type_path]
#         dataset = self.dataset_class(
#             self.tokenizer,
#             type_path=type_path,
#             n_obs=n_obs,
#             max_target_length=max_target_length,
#             **self.dataset_kwargs,
#         )
#         return dataset

#     def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
#         dataset = self.get_dataset(type_path)

#         if self.hparams.sortish_sampler and type_path != "test":
#             sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
#             return DataLoader(
#                 dataset,
#                 batch_size=batch_size,
#                 collate_fn=dataset.collate_fn,
#                 shuffle=False,
#                 num_workers=self.num_workers,
#                 sampler=sampler,
#             )

#         elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
#             batch_sampler = dataset.make_dynamic_sampler(
#                 self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
#             )
#             return DataLoader(
#                 dataset,
#                 batch_sampler=batch_sampler,
#                 collate_fn=dataset.collate_fn,
#                 # shuffle=False,
#                 num_workers=self.num_workers,
#                 # batch_size=None,
#             )
#         else:
#             return DataLoader(
#                 dataset,
#                 batch_size=batch_size,
#                 collate_fn=dataset.collate_fn,
#                 shuffle=shuffle,
#                 num_workers=self.num_workers,
#                 sampler=None,
#             )

#     def train_dataloader(self) -> DataLoader:
#         dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
#         return dataloader

#     def val_dataloader(self) -> DataLoader:
#         return self.get_dataloader("val", batch_size=self.hparams.dev_batch_size)

#     def test_dataloader(self) -> DataLoader:
#         return self.get_dataloader("test", batch_size=self.hparams.dev_batch_size)

#     @staticmethod
#     def add_model_specific_args(parser, root_dir):
#         BaseTransformer.add_model_specific_args(parser, root_dir)
#         add_generic_args(parser, root_dir)
#         parser.add_argument(
#             "--max_source_length",
#             default=1024,
#             type=int,
#             help="The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated, sequences shorter will be padded.",
#         )
#         parser.add_argument(
#             "--max_target_length",
#             default=100,
#             type=int,
#             help="The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated, sequences shorter will be padded.",
#         )
#         parser.add_argument(
#             "--val_max_target_length",
#             default=100,  # these defaults are optimized for CNNDM. For xsum, see README.md.
#             type=int,
#             help="The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated, sequences shorter will be padded.",
#         )
#         parser.add_argument(
#             "--test_max_target_length",
#             default=100,
#             type=int,
#             help="The maximum total input sequence length after tokenization. Sequences longer "
#             "than this will be truncated, sequences shorter will be padded.",
#         )
#         parser.add_argument("--freeze_encoder", action="store_true")
#         parser.add_argument("--freeze_embeds", action="store_true")
#         parser.add_argument("--sortish_sampler", action="store_true", default=False)
#         parser.add_argument("--max_tokens_per_batch", type=int, default=None)
#         parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
#         parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
#         parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
#         parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
#         parser.add_argument(
#             "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
#         )
#         parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
#         parser.add_argument("--src_lang", type=str, default="", required=False)
#         parser.add_argument("--tgt_lang", type=str, default="", required=False)
#         parser.add_argument("--eval_beams", type=int, default=None, required=False)
#         parser.add_argument(
#             "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
#         )
#         parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
#         parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
#         parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
#         parser.add_argument(
#             "--early_stopping_patience",
#             type=int,
#             default=-1,
#             required=False,
#             help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
#         )
#         return parser

class PrefixDS2(PrefixTransformer):
    mode = "summarization"
    # loss_names = ["loss"]
    # metric_names = ROUGE_KEYS
    # default_val_metric = "rouge2"

    def __init__(self, hparams, qa_model, **kwargs):
        # if args.sortish_sampler and args.gpus > 1: # default = None
        #     args.replace_sampler_ddp = False
        # elif args.max_tokens_per_batch is not None: # default = None
        #     if args.gpus > 1:
        #         raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
        #     if args.sortish_sampler:
        #         raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        super().__init__(hparams, num_labels=None, mode= self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")
        # save_git_info(self.hparams.output_dir)
        # self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        # self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        # pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type # e.g.) bart
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size
        
        # if self.hparams.freeze_embeds:
        #     self.freeze_embeds()
        # In init, freeze all parameters of the seq2seq_model (Bart)
        freeze_params(self.seq2seq_model)
        assert_all_frozen(self.seq2seq_model)
        print('FREEZING ENTIRE seq2seq model.')
        # if self.hparams.freeze_encoder:
        #     freeze_params(self.model.get_encoder())
        #     assert_all_frozen(self.model.get_encoder())

        # set decoder_start_token_id
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id

        # self.hparams = hparams -> Attribute Error
        self.save_hyperparameters(hparams)
        
        # self.tokenizer = tokenizer

        # self.sum_model = self.model
        if self.hparams["use_qa_deconverter"]:
            self.qa_model = qa_model # None
        self.lr = self.hparams["lr"]
        self.blank = "____"

        # converter
        self.converter = get_converter(self.hparams['state_converter'])
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

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        if self.model_type == "t5":
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)
        elif self.model_type == "fsmt":
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        else: # e.g.) bart
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, gpt2_model=self.seq2seq_model, **kwargs)  

    # token ids -> decoded sentence
    def ids_to_clean_text(self, generated_ids: List[int]):
        # batch_decode: convert a list of lists of token ids into a list of strings by calling decode.
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    # basic step
    def _step(self, batch: dict): #-> Tuple:
        pad_token_id = self.tokenizer.pad_token_id

        src_ids, src_mask = batch["encoder_input"], batch["attention_mask"]
        tgt_ids = batch["decoder_output"]

        # "shift_right": is used if input_ids and labels are provided, but no decoder_input_ids.
        # In this case this function automatically creates the correct decoder_input_ids
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        
        # forward
        # use_cache: used for generation, not for training 
        outputs = self(src_ids, attention_mask = src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, use_prefix=True)

        lm_logits = outputs[0]

        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            # print(lm_logits.shape, tgt_ids.shape, lm_logits.shape[-1] )
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else: # label_smoothing
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )

        # return (loss,)
        return Seq2SeqLMOutput(loss)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx):
        self.model.train()

        # outputs = self.sum_model(
        #     input_ids=batch["encoder_input"],
        #     attention_mask=batch["attention_mask"],
        #     labels=batch["decoder_output"],
        # )

        outputs = self._step(batch)

        return {'loss': outputs.loss, 'log': {'train_loss': outputs.loss.detach()}}

    # evaluate for each step
    def eval_step(self, batch, batch_idx):
        self.model.eval()

        # outputs = self.sum_model(
        #     input_ids=batch["encoder_input"],
        #     attention_mask=batch["attention_mask"],
        #     labels=batch["decoder_output"],
        # )

        outputs = self._step(batch)

        return outputs.loss.item()

    # generate with prompt
    def pred_step(self, batch: dict,batch_idx): # -> dict:
        self.model.eval()

        # write the prompt generation from self.model.
        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        # get the prompt:
        bsz = batch["encoder_input"].size(0)
        # use num_beams = 1 from DS2
        prefix_prompt = self.model.get_prompt(bsz=bsz, sample_size=self.hparams["num_beams"]) 
        # print(prefix_prompt)
        generated_ids = self.seq2seq_model.generate(
            batch["encoder_input"],
            past_key_values=prefix_prompt,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            length_penalty=self.hparams.length_penalty,
            use_prefix=True,
            decoder_start_token_id=self.decoder_start_token_id,
            # below, same values as DS2
            num_beams=self.hparams["num_beams"],
            min_length=5,
            max_length=100,
            early_stopping=True
        )

        return {
            "pred_summary_token": generated_ids,
            "gold_state": batch["slot_values"],
            "gold_summary": batch["output_text"],
            "eval_slots": batch["eval_slots"],
        }

    def eval_epoch_end(self, outputs):
        res = {}
        res["loss"] = np.mean(outputs)
        print(res)
        return res

    # Make gold template from y:states and compare with predicted summarys
    def pred_epoch_end(self, outputs, mode="val"):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}

        # pred_summary : tokenized predicted summary
        pred_summary = [
            self.tokenizer.decode(_sum, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _sum in outputs["pred_summary_token"]
        ]

        # qa_model = None
        if self.hparams["use_qa_deconverter"]:
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

        # compare pred_summary and gold template(In training)
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
    def validation_step(self, batch, batch_idx):
        if self.hparams["eval_loss_only"]: 
            return self.eval_step(batch, batch_idx)
        else:
            return self.pred_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if self.hparams["eval_loss_only"]:
            res = {f'val_{k}': v for k, v in self.eval_epoch_end(outputs).items()}
        else:
            res = {f'val_{k}': v for k, v in self.pred_epoch_end(outputs, "val").items()}
        self.log_dict(res)
        return res

    """ test """
    def test_step(self, batch, batch_idx):
        return self.pred_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        res = {f'test_{k}': v for k, v in self.pred_epoch_end(outputs, "test").items()}
        self.log_dict(res)
        return res

    # optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def save_samples(self, samples, name):
        if self.hparams["save_samples"] > 0:
            output_fields = ['only_domain', 'fewshot', 'grad_acc_steps', 'train_batch_size', 'state_converter']
            output_name = '_'.join([str(self.hparams[k]) for k in output_fields]) + '_' + name + '_' + str(round(time.time()))
            filename = f'./samples_data/{output_name}.json'
            with open(filename, 'w') as f:
                json.dump({k: v[:self.hparams['save_samples']] for k, v in samples.items()}, f)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        PrefixTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=512, #1024
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56, #56
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  #142 # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142, #142
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task_mode", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--length_penalty", type=float, default=1.0, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser

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