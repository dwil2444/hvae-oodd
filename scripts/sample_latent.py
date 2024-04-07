"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import os
import logging

from collections import defaultdict
from typing import *

from tqdm import tqdm

import rich
import numpy as np
import torch

import oodd
import oodd.datasets
import oodd.evaluators
import oodd.models
import oodd.losses
import oodd.utils
# from .ood_llr import get_decode_from_p

LOGGER = logging.getLogger()


def get_decode_from_p(n_latents, k=0, semantic_k=True):
    """
    k semantic out
    0 True     [False, False, False]
    1 True     [True, False, False]
    2 True     [True, True, False]
    0 False    [True, True, True]
    1 False    [False, True, True]
    2 False    [False, False, True]
    """
    if semantic_k:
        return [True] * k + [False] * (n_latents - k)

    return [False] * (k + 1) + [True] * (n_latents - k - 1)


@torch.no_grad
def main():
    device = oodd.utils.get_device() if args.device == "auto" else torch.device(args.device)
    checkpoint = oodd.models.Checkpoint(path=args.model_dir)
    checkpoint.load()
    datamodule = checkpoint.datamodule
    model = checkpoint.model
    model.eval()
    #rich.print(datamodule)
    LOGGER.info(datamodule.train_loaders.items())
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models/FashionMNIST", help="model")
    parser.add_argument("--iw_samples_elbo", type=int, default=1, help="importances samples for regular ELBO")
    parser.add_argument("--iw_samples_Lk", type=int, default=1, help="importances samples for L>k bound")
    parser.add_argument("--n_latents_skip", type=int, default=1, help="the value of k in the paper")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="device to evaluate on")
    args = parser.parse_args()
    main()
