import os
import pandas as pd

import jax
import torchvision

import reprieve
from reprieve.representations import mnist_vae
from reprieve.mnist_noisy_label import MNISTNoisyLabelDataset
from reprieve.algorithms import mlp as alg

from argparse import Namespace
from functools import reduce
import numpy as np
import torch
from torch.utils.data import Dataset

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
targetdir = os.path.dirname(currentdir) + '/reprieve'
sys.path.insert(0, targetdir)

import dataset_wrappers
import utils

class SpectralDataset(Dataset):
    """Dataset with different frequencies."""

    def __init__(self, t, yt, transform=None, binary=False):
        """
        Args:
            t: inputs
            yt: targets
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.t = torch.from_numpy(t).view(-1, 1)
        if binary:
            self.yt = torch.from_numpy(yt>0).int()
        else:
            self.yt = torch.from_numpy(yt)
        self.transform = transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        sample = (self.t[idx], self.yt[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    # Data Generation with constant amplitude of 1 for all frequencies
    opt = Namespace()
    opt.N = 1000
    opt.K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    opt.A = [1 for _ in opt.K]
    opt.PHI = [np.random.rand() for _ in opt.K]


    def make_phased_waves(opt):
        t = np.arange(0, 1, 1. / opt.N)
        if opt.A is None:
            yt = reduce(lambda a, b: a + b,
                        [np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, phi in zip(opt.K, opt.PHI)])
        else:
            yt = reduce(lambda a, b: a + b,
                        # [Ai * np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])
                        [Ai * np.sin(2 * np.pi * ki * t + phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])
        return t, yt

    t, yt = make_phased_waves(opt)
    dataset_spectral = SpectralDataset(t, yt)
    batch_size = 256
    product_seeds = [11]
    product_points = [200]

    results_list = []
    for depth in range(1, 4):
        init_fn, train_step_fn, eval_fn = alg.make_algorithm(input_shape=(1,), n_classes=1, n_layers=depth, h_dim=10)

        # these functions initialize the state of the model given a seed,
        # train on a batch of data,
        # and evaluate on a batch of data

        # construct a loss-data estimator and use it to compute the loss-data curve
        loss_data_estimator = reprieve.LossDataEstimator(
            init_fn, train_step_fn, eval_fn, dataset_spectral)

        # Use _train
        train_set = dataset_wrappers.DatasetSubset(
            dataset_spectral, stop=int(0.9*len(dataset_spectral)))
        state = loss_data_estimator._train(seed=11, dataset=train_set)
        train_set = utils.dataset_to_jax(
            train_set,
            batch_transforms=[lambda x: x],
            batch_size=batch_size)
        preds = state.target(train_set[0])

        results = loss_data_estimator.compute_curve()
        # name experiments
        results['name'] = f'Layers-{depth}'
        results_list.append(results)