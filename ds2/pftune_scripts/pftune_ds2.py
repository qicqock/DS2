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

from ds2.pftune_helper.pftune_config import get_args
from ds2.pftune_models.pftune_model import PrefixSummarizationModule, PrefixDS2
from ds2.pftune_models.lightning_base import OurModelCheckPoint

logger = logging.getLogger(__name__)

def prefix_tune(args_ns, *more):
    # args_ns is the same as args except for the object type
    # args_ns: Namespace object. ex) args.model_checkpoint
    # args: dictionary object. ex) args["model_checkpoint"]
    args = vars(args_ns)
    seed_everything(args["seed"])
    pl.seed_everything(args["seed"]) # seed lightning
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args["model_checkpoint"])
    # model = AutoModelForSeq2SeqLM.from_pretrained(args["model_checkpoint"])

    # load data from train_dial.json, valid_dial.json, test_dial.json
    dataloaders, _ = prepare_data(
        args, tokenizer
    )
    print("Created dataloaders")

    # mkdir to save model
    exp_name = args["exp_name"]
    if not os.path.exists("ds2/logs"):
        os.mkdir("ds2/logs")
    log_path = f"ds2/logs/{exp_name}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print("save_path is  {}".format(log_path))

    # load pretrained language model
    # determine whether we pre-train or fine-tune
    if args["load_pretrained"]:
        pretrain_ckpt_path = os.path.join(args["load_pretrained"], "pretrain")
        pretrain_ckpts = [
            _ckpt for _ckpt in os.listdir(pretrain_ckpt_path)
            if ".ckpt" in _ckpt
        ]
        assert len(pretrain_ckpts) == 1
        ckpt = pretrain_ckpts[0]
        print("load pretrained model from: ", os.path.join(pretrain_ckpt_path, ckpt))
        dst_model = DS2.load_from_checkpoint(
            os.path.join(pretrain_ckpt_path, ckpt),
            args=args,
            tokenizer=tokenizer,
            sum_model=model,
            qa_model=None,
        )
    else:
        dst_model = PrefixDS2(args, None)

    print("Created Model")

    # determine the path
    dir_path = os.path.join(log_path, args["mode"], str(args["seed"]))

    # if train, do earlystopping 
    if not args["do_test_only"]:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss" if args["eval_loss_only"] else "val_jga",
            # min_delta=0.00,
            patience=args["patience"],
            verbose=False,
            mode="min" if args["eval_loss_only"] else "max",
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(dir_path, "{epoch}-{val_loss:.4f}-{val_jga:.4f}"), 
            save_top_k=1,
            monitor="val_loss" if args["eval_loss_only"] else "val_jga",
            mode="min" if args["eval_loss_only"] else "max",
        )
        # checkpoint_callback = OurModelCheckPoint(
        #     # In lower version of pytorch_lightning, filepath instead of filename and dirpath
        #     filepath=os.path.join(dir_path, "{epoch}-{val_loss:.3f}-{val_jga:.3f}"), 
        #     save_top_k=1,
        #     monitor="val_loss" if args["eval_loss_only"] else "val_jga",
        #     mode="min" if args["eval_loss_only"] else "max",
        # )

        callbacks = [earlystopping_callback, checkpoint_callback]
    else:
        callbacks = None
        earlystopping_callback = None
        checkpoint_callback = None

    # profiler = PyTorchProfiler(export_to_chrome=True)
    trainer = Trainer(
        accumulate_grad_batches=args["grad_acc_steps"], 
        gradient_clip_val=args["max_norm"],
        max_epochs=args["n_epochs"],
        # callbacks=callbacks,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=earlystopping_callback,
        gpus=args["GPU"], 
        deterministic=True,
        # accelerator="ddp", # multiGPU
        val_check_interval=args["val_check_interval"],
        # logger=CSVLogger(dir_path, f"seed_{args['seed']}") if not args["do_test_only"] else None,
        logger=CSVLogger(dir_path) if not args["do_test_only"] else None,
        resume_from_checkpoint=args["resume_from_ckpt"],
        # limit_val_batches=0.05,
        # # set accelrator to auto
        # accelerator="ddp2"
    )

    # if train, fit the trainer
    if not args["do_test_only"]:
        trainer.fit(dst_model, dataloaders["train"], dataloaders["dev"])

if __name__ == "__main__":
    args = get_args()
    prefix_tune(args)
