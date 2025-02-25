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
    'method': 'bayes'
    }

sweep_config['metric'] = {
    'name': 'loss',
    'goal': 'minimize'   
    }

parameters_dict = {
    'optimizer': {
        'values': ['Adam', 'RMSprop'],
        },
    'fc_layers': {
        'values': [[256], [256, 256]]
        },
    'fc_dropout': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
    'num_channels': {
        'values': [[256, 128, 64]]
    },
    'conv_dropout': {
        'values': [0.1, 0.15, 0.2]
    },
    'batch_size': {
        'values': [64, 128]
        },
    # 'loss_fn': {
    #     'values': ['MSELoss']
    # },
    'loss_fn': {
        'values': ['ELBOLoss']
    },
    'lr': {
        'values': [0.01, 0.001]
    }
    }

sweep_config['parameters'] = parameters_dict

# Define the sweep configuration
sweep_id = wandb.sweep(sweep_config, project='Bayes-autoencoder')

# Fixed hyperparameters
network_name = 'CNN-VAE'
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
        fc_dropout = config.fc_dropout
        conv_dropout = config.conv_dropout
        lr = config.lr
        batch_size = config.batch_size
        loss_fn_name = config.loss_fn

        # Update the model and optimizer with the parameters from the sweep
        model = NET_TYPES[network_name](seq_len=seq_len, latent_dim=dim, fc_dropout=fc_dropout, conv_dropout=conv_dropout)
        loss_fn = LOSS_FUNCTIONS[loss_fn_name]()
        optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay)

        # Dataloaders
        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        best_loss = float("inf")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for seqs in train_loader:
                if use_cuda:
                    seqs = seqs.cuda()
                    model.cuda()

                seqs = seqs.float()
                optimizer.zero_grad()
                outputs = model(seqs)
                loss = loss_fn(outputs, seqs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation loop
            model.eval()
            valid_loss = 0.0
            reconstruction_acc = 0.0
            cosine_sim = 0.0
            pearson_corr = 0.0
            num_batches = 0

            with torch.no_grad():
                for seqs in valid_loader:
                    if use_cuda:
                        seqs = seqs.cuda()
                    seqs = seqs.float()
                    outputs = model(seqs)
                    loss = loss_fn(outputs, seqs)
                    valid_loss += loss.item()

                    # Compute additional metrics
                    reconstruction_acc += reconstruction_accuracy(seqs, outputs)
                    cosine_sim += cosine_similarity(seqs, outputs)
                    pearson_corr += pearson_correlation(seqs, outputs)
                    num_batches += 1

            # Average metrics
            valid_loss /= len(valid_loader)
            reconstruction_acc /= num_batches
            cosine_sim /= num_batches
            pearson_corr /= num_batches

            # Log metrics to W&B
            wandb.log({
                'train_loss': train_loss / len(train_loader),
                'valid_loss': valid_loss,
                'reconstruction_accuracy': reconstruction_acc,
                'cosine_similarity': cosine_sim,
                'pearson_correlation': pearson_corr,
            })

            # Save model if it is the last epoch or the reconstruction loss is lower than the best one so far
            if epoch == num_epochs:
                torch.save(model.state_dict(), os.path.join(args.output, '{}_last.model'.format(namespace)))

            if valid_loss < best_loss and epoch < num_epochs:
                torch.save(model.state_dict(), os.path.join(args.output, "{}_{}.model".format(namespace, epoch + 1)))
                best_loss = valid_loss


# Launch the sweep agent
wandb.agent(sweep_id, function=train_model)
