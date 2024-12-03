import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import math
import os
import numpy as np
from time import time
from datetime import datetime

from bin.datasets import SeqDataset
from bin.common import *
from bin.models import *
from argparser import *

import wandb
wandb.login()

sweep_config = {
    'method': 'random'
    }
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }
sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam']
        },
    'fc_layers': {
        'values': [[], []]
        },
    'fc_dropout': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
    'num_channels': {
        'values': [[], []]
    },
    'conv_dropout': {
        'values': [0.1, 0.15, 0.2]
    },
    'batch_size': {
        'values': [64, 128]
        },
    'loss_fn': {
        'values': ['mse', 'cross-enthropy']
    },
    'lr': {
        'values': [0.001]
    }
    }

sweep_config['parameters'] = parameters_dict

# Define the sweep configuration
sweep_id = wandb.sweep(sweep_config, project='autoencoder')

# Fixed hyperparameters
network_name = 'CNN'
noise = False
seq_len = args.seq_len
dim = args.dim
num_epochs = args.num_epochs
weight_decay = args.weight_decay
namespace = args.namespace

# Creating a dataset
dataset = SeqDataset(filename=args.path+"/"+args.filename, noise=noise, seq_len=seq_len)
train_ids, valid_ids, test_ids = dataset.train_ids, dataset.valid_ids, dataset.test_ids
indices = [train_ids, valid_ids, test_ids]

# Check device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def train_model():
    # Initialize W&B run
    with wandb.init() as run:
        config = wandb.config

        # Use the hyperparameters from the sweep config
        optimizer_name = config.optimizer
        dropout_fc = config.fc_dropout
        dropout_conv = config.conv_dropout
        lr = config.lr
        batch_size = config.batch_size
        loss_fn_name = config.loss_fn

        # Update the model and optimizer with the parameters from the sweep
        network = NET_TYPES[network_name](seq_len=seq_len, latent_dim=dim, dropout_fc=dropout_fc, dropout_conv=dropout_conv)
        loss_fn = LOSS_FUNCTIONS[loss_fn_name]()
        optimizer = OPTIMIZERS[optimizer_name](network.parameters(), lr=lr, weight_decay=weight_decay)

        # Dataloaders
        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        best_loss = float("inf")

        for epoch in range(num_epochs):
            network.train()
            train_loss = 0.0
            for seqs in train_loader:
                if use_cuda:
                    seqs = seqs.cuda()
                    network.cuda()

                seqs = seqs.float()
                optimizer.zero_grad()
                outputs = network(seqs)
                loss = loss_fn(outputs, seqs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation loop
            network.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for seqs in valid_loader:
                    if use_cuda:
                        seqs = seqs.cuda()
                    seqs = seqs.float()
                    outputs = network(seqs)
                    loss = loss_fn(outputs, seqs)
                    valid_loss += loss.item()

            # Log metrics to W&B
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'valid_loss': valid_loss / len(valid_loader),
            })

            # Save the best model
            if valid_loss < best_loss:
                torch.save(network.state_dict(), os.path.join(args.output, f"{namespace}_best.model"))
                best_loss = valid_loss

# Launch the sweep agent
wandb.agent(sweep_id, function=train_model)
