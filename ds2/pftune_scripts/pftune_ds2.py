# ds2 dependent
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from ds2.datasets.data_loader import (
    prepare_data,
)
# from ds2.configs.config import get_args
from ds2.models.DS2 import DS2

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

# etc
from ds2.pftune_helper.pftune_config import get_args
from ds2.pftune_models.pftune_model import PrefixSummarizationModule, SummarizationModule

logger = logging.getLogger(__name__)

def prefix_tune(args_ns, *more):
    # args_ns is the same as args except for the object type
    # args_ns: Namespace object. ex) args.model_checkpoint
    # args: dictionary object. ex) args["model_checkpoint"]
    args = vars(args_ns)
    seed_everything(args["seed"])
    print(args)

    # # load data from train_dial.json, valid_dial.json, test_dial.json
    # dataloaders, _ = prepare_data(
    #     args, tokenizer
    # )
    # print("Created dataloaders")

    # # mkdir to save model
    # exp_name = args["exp_name"]
    # if not os.path.exists("ds2/logs"):
    #     os.mkdir("ds2/logs")
    # log_path = f"ds2/logs/{exp_name}"
    # if not os.path.exists(log_path):
    #     os.mkdir(log_path)
    # print("save_path is  {}".format(log_path))

    # # load pretrained language model
    # # determine whether we pre-train or fine-tune
    # if args["load_pretrained"]:
    #     pretrain_ckpt_path = os.path.join(args["load_pretrained"], "pretrain")
    #     pretrain_ckpts = [
    #         _ckpt for _ckpt in os.listdir(pretrain_ckpt_path)
    #         if ".ckpt" in _ckpt
    #     ]
    #     assert len(pretrain_ckpts) == 1
    #     ckpt = pretrain_ckpts[0]
    #     print("load pretrained model from: ", os.path.join(pretrain_ckpt_path, ckpt))
    #     dst_model = DS2.load_from_checkpoint(
    #         os.path.join(pretrain_ckpt_path, ckpt),
    #         args=args,
    #         tokenizer=tokenizer,
    #         sum_model=model,
    #         qa_model=None,
    #     )
    # else:
    #     dst_model = DS2(args, tokenizer, model, None)

    # seed lightning
    pl.seed_everything(args["seed"])

    # model init
    model = None
    if args["tuning_mode"] == 'prefixtune':
        model = PrefixSummarizationModule(args_ns)
    # elif args.tuning_mode == 'finetune':
    #     model: SummarizationModule = SummarizationModule()
    else:
        assert False, 'invalid tuning_mode'
    print("Created Model")

    print("time_sleep 100")
    time.sleep(100)

    # trainer
    lower_is_better = args.val_metric == "loss" # default = 0
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric, args.save_top_k, lower_is_better), #LISA
        early_stopping_callback=es_callback,
        logger=logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test()


    # # determine the path
    # dir_path = os.path.join(log_path, args["mode"])
    # # if do not test(train), do earlystopping 
    # if not args["do_test_only"]:
    #     earlystopping_callback = EarlyStopping(
    #         monitor="val_loss" if args["eval_loss_only"] else "val_jga",
    #         # min_delta=0.00,
    #         patience=args["patience"],
    #         verbose=False,
    #         mode="min" if args["eval_loss_only"] else "max",
    #     )
    #     checkpoint_callback = ModelCheckpoint(
    #         dirpath=dir_path,
    #         filename="{val_loss:.3f}" if args["eval_loss_only"] else "{val_jga:.3f}",
    #         save_top_k=1,
    #         monitor="val_loss" if args["eval_loss_only"] else "val_jga",
    #         mode="min" if args["eval_loss_only"] else "max",
    #     )
    #     callbacks = [earlystopping_callback, checkpoint_callback]
    # else:
    #     callbacks = None

    # # profiler = PyTorchProfiler(export_to_chrome=True)
    # trainer = Trainer(
    #     accumulate_grad_batches=args["grad_acc_steps"], 
    #     gradient_clip_val=args["max_norm"],
    #     max_epochs=args["n_epochs"],
    #     callbacks=callbacks, # earlystopping
    #     gpus=args["GPU"], 
    #     deterministic=True,
    #     accelerator="ddp", # multiGPU
    #     val_check_interval=args["val_check_interval"],
    #     logger=CSVLogger(dir_path, f"seed_{args['seed']}") if not args["do_test_only"] else None,
    #     resume_from_checkpoint=args["resume_from_ckpt"],
    #     # limit_val_batches=0.05,
    #     # # set accelrator to auto
    #     # accelerator="ddp2"
    # )


    # # if train, fit the trainer
    # if not args["do_test_only"]:
    #     trainer.fit(dst_model, dataloaders["train"], dataloaders["dev"])

    # # if test
    # if not args["do_train_only"]:
    #     print("test start...")
    #     # evaluate model
    #     args["num_beams"] = args["test_num_beams"]
    #     if args["do_test_only"]:
    #         ckpts = [_ckpt for _ckpt in os.listdir(dir_path) if ".ckpt" in _ckpt]
    #         assert len(ckpts) == 1
    #         ckpt = ckpts[0]
    #         print("load pretrained model from: ", os.path.join(dir_path, ckpt))
    #         ckpt_path = os.path.join(dir_path, ckpt)
    #     else:
    #         ckpt_path = checkpoint_callback.best_model_path

    #     # load trained model from checkpoint
    #     dst_model = DS2.load_from_checkpoint(
    #         checkpoint_path=ckpt_path,
    #         args=args,
    #         tokenizer=tokenizer,
    #         sum_model=model,
    #         qa_model=None
    #     )
    #     trainer.test(dst_model, dataloaders["test"])


if __name__ == "__main__":
    args = get_args()
    prefix_tune(args)
