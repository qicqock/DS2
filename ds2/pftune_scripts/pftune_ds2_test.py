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

def prefix_tune_test(args_ns, *more):
    # args_ns is the same as args except for the object type
    # args_ns: Namespace object. ex) args.model_checkpoint
    # args: dictionary object. ex) args["model_checkpoint"]
    args = vars(args_ns)
    seed_everything(args["seed"])
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

    # determine the path
    dir_path = os.path.join(log_path, args["mode"], str(args["seed"]))

    callbacks = None
    earlystopping_callback = None
    checkpoint_callback = None

    # profiler = PyTorchProfiler(export_to_chrome=True)
    trainer = Trainer(
        accumulate_grad_batches=args["grad_acc_steps"], 
        gradient_clip_val=args["max_norm"],
        max_epochs=args["n_epochs"],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=earlystopping_callback,
        gpus=args["GPU"], 
        deterministic=True,
        val_check_interval=args["val_check_interval"],
        logger=None,
        resume_from_checkpoint=args["resume_from_ckpt"],
        # limit_val_batches=0.05,
        # # set accelrator to auto
        # accelerator="ddp2"
    )

    print("test start...")

    # evaluate model
    args["num_beams"] = args["test_num_beams"]

    dst_model = PrefixDS2(args, None)
    trainer.test(dst_model, dataloaders["test"])

if __name__ == "__main__":
    args = get_args()
    prefix_tune_test(args)
