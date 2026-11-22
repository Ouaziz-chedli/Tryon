"""Shared helpers for the Track 1 U-Net scripts."""
import numpy as np  # noqa: F401
import torch
import torch.nn as nn


def blk(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def to_t(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255)


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2))
    den = p.sum((1, 2)) + y.sum((1, 2))
    return (1 - (2 * inter + 1) / (den + 1)).mean()
