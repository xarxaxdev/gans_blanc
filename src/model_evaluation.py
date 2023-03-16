### Here will be the main functions to evaluate the models
### Maybe draw some plots in src/plots

import torch
from torchmetrics.classification import MulticlassF1Score
from model_generation import ent_to_ix

def compute_f1(prediction, target):
    metric = MulticlassF1Score(num_classes=len(ent_to_ix))
    return metric(prediction, target)
