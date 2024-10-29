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

# TODO: poprawić results_header, validate
    # loss_fn poprawić
    # dodać argument seq_len
    # dodać do loggera info o rozmiarze latent_dim


seq_len = 200

batch_size, num_workers, num_epochs, acc_threshold, seq_len, namespace = args.batch_size, args.num_workers, args.num_epochs, args.acc_threshold, args.seq_len, args.namespace
dim = args.dim

# Set seed
seed=args.seed
torch.manual_seed(seed)  
np.random.seed(seed)

network_name = "basic"
if network_name.lower() == "basic":
    RESULTS_COLS = OrderedDict({'Loss': ['losses', 'float-list']})

optimizer_name = args.optimizer
lossfn_name = args.loss_fn
weight_decay = args.weight_decay
momentum = args.momentum

network = NET_TYPES[network_name.lower()]
optim_method = OPTIMIZERS[optimizer_name]
lossfn = LOSS_FUNCTIONS[lossfn_name]

lr = args.learning_rate
adjust_lr = False if args.no_adjust_lr else True

modelfile = args.model if os.path.isfile(args.model) else None


# Define files for logs and for results
[logger, results_table], old_results = build_loggers('train', output=args.output, namespace=namespace)

logger.info('\nAnalysis {} begins {}\nInput data: {}\nOutput directory: {}\n'.format(
    namespace, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), args.path, args.output))

t0 = time()
# CUDA for PyTorch
use_cuda, device = check_cuda(logger)
noise = not args.no_noise

dataset = SeqDataset(filename=args.path+"/"+args.filename, noise=noise, seq_len=seq_len)

# Write header of results table
if not old_results:
    results_table, columns = results_header('train', results_table, RESULTS_COLS)
else:
    columns = read_results_columns(results_table, RESULTS_COLS)

# Creating data indices for training, validation and test splits:
train_ids, valid_ids, test_ids = dataset.train_ids, dataset.valid_ids, dataset.test_ids
indices = [train_ids, valid_ids, test_ids]


# class_stage = [dataset.get_classes(el) for el in indices]
train_len, valid_len = len(train_ids), len(valid_ids)

num_seqs = ' + '.join([str(len(el)) for el in [train_ids, valid_ids, test_ids]])
chr_string = ['', '', '']
for i, (n, ch, ind) in enumerate(zip(['train', 'valid', 'test'], chr_string,
                                     [train_ids, valid_ids, test_ids])):
    logger.info('\n{} set contains {} seqs {}:'.format(n, len(ind), ch))

    # Writing IDs for each split into file
    with open(os.path.join(args.output, '{}_{}.txt'.format(namespace, n)), 'w') as f:
        f.write('\n'.join([dataset.info[j] for j in ind]))

train_sampler = SubsetRandomSampler(train_ids)
valid_sampler = SubsetRandomSampler(valid_ids)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

logger.info('\nTraining and validation datasets built in {:.2f} s'.format(time() - t0))


num_batches = math.ceil(train_len / batch_size)
model = network(seq_len=dataset.seq_len, latent_dim=dim)

if modelfile is not None:
    t0 = time()
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device)), strict=False)
    logger.info('\nModel from {} loaded in {:.2f} s'.format(modelfile, time() - t0))

if args.dropout_fc is not None:
    network.dropout_fc = args.dropout_fc
    logger.info('\nDropout-fc changed to {}'.format(args.dropout_fc))
if args.dropout_conv is not None:
    network.dropout_conv = args.dropout_conv
    logger.info('\nDropout-conv changed to {}'.format(args.dropout_conv))

if optimizer_name == 'RMSprop':
    optimizer = optim_method(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
else:
    optimizer = optim_method(model.parameters(), lr=lr, weight_decay=weight_decay)

# loss_fn = lossfn()
loss_fn = nn.MSELoss()
best_acc = 0.0
best_loss = math.inf


# TRAINING LOOP

logger.info('\n--- TRAINING ---\nEpoch 0 is a data validation without training step')
t = time()
for epoch in range(num_epochs + 1):
    t0 = time()
    train_loss = 0.0
    valid_loss = 0.0
    train_correct_bases = 0
    valid_correct_bases = 0
    train_total_bases = 0
    valid_total_bases = 0
    
    if epoch == num_epochs:
        # If this is the last epoch, store outputs for inspection
        train_output_values = [[] for _ in range(seq_len)]
        valid_output_values = [[] for _ in range(seq_len)]
    
    # Training loop
    model.train() 
    for i, seqs in enumerate(train_loader):
        if use_cuda:
            seqs = seqs.cuda()
            model.cuda()

        seqs = seqs.float()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(seqs)

        loss = loss_fn(outputs, seqs)
        loss.backward()  # Calculate gradients
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # # Print gradients for each parameter
        # for name, param in model.named_parameters():
        #     if param.grad is not None:  # Check if gradients exist
        #         print(f'Gradient for {name}: {param.grad.data}')

        optimizer.step()

        # Track total loss for the epoch
        train_loss += loss.item()

        # Calculate base-level accuracy
        # correct = calculate_accuracy(outputs, seqs)

        # [64, 1, 4, 200]
        _, predicted = torch.max(outputs, dim=2)  # [64, 1, 200]
        _, true = torch.max(seqs, dim=2)
        correct = (predicted == true).sum().item()
        print(f'correct: {correct}')
        train_correct_bases += correct  # Total correct bases in this batch
        train_total_bases += seqs.size(0) * seqs.size(3)  # Total bases in this batch

        print(train_correct_bases, '/', train_total_bases)

        if epoch == num_epochs:
            # Store neuron outputs for the last epoch for analysis
            for j, outp in enumerate(outputs):
                train_output_values[j].append(outp.detach().cpu().numpy())

        if i % 10 == 0:
            logger.info('Epoch {}, batch {}/{}'.format(epoch, i, num_batches))
    
    # Validation loop 
    model.eval()
    with torch.no_grad():
        for i, seqs in enumerate(valid_loader):
            if use_cuda:
                seqs = seqs.cuda()

            seqs = seqs.float()

            # Forward pass
            outputs = model(seqs)
            loss = loss_fn(outputs, seqs)  # Compute validation loss

            valid_loss += loss.item()
            _, predicted = torch.max(outputs, dim=2)  
            _, true = torch.max(seqs, dim=2)
            correct = (predicted == true).sum().item()
            valid_correct_bases += correct  # Total correct bases in this batch
            valid_total_bases += seqs.size(0) * seqs.size(3)  # Total bases in this batch

            if epoch == num_epochs:
                # Store neuron outputs for the last epoch for analysis
                for j, outp in enumerate(outputs):
                    valid_output_values[j].append(outp.detach().cpu().numpy())

    # Adjust the learning rate if necessary
    if not args.no_adjust_lr:
        adjust_learning_rate(lr, epoch, optimizer)

    # Calculate and log the mean loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    avg_valid_loss = valid_loss / len(valid_loader)

    train_accuracy = train_correct_bases / train_total_bases
    valid_accuracy = valid_correct_bases / valid_total_bases

    # Store the loss values in a format expected by `write_results`
    globals()['train_losses'] = [avg_train_loss]
    globals()['valid_losses'] = [avg_valid_loss]
    globals()['train_accuracy'] = train_accuracy
    globals()['valid_accuracy'] = valid_accuracy


    # If it's the last epoch, save outputs for analysis
    if epoch == num_epochs:
        logger.info('Last epoch - saving outputs!')
        # Save the output values as object arrays
        np.save(os.path.join(args.output, '{}_train_outputs'.format(namespace)), np.array(train_output_values, dtype=object))
        np.save(os.path.join(args.output, '{}_valid_outputs'.format(namespace)), np.array(valid_output_values, dtype=object))
        torch.save(model.state_dict(), os.path.join(args.output, '{}_last.model'.format(namespace)))

    # Write the results
    write_results(results_table, columns, ['train', 'valid'], globals(), epoch)
    logger.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}\tTrain accuracy: {:1.3f}\nValid loss: {:1.3f}\tValid accuracy: {:1.3f}".format(
        epoch, (time() - t0) / 60, avg_train_loss, train_accuracy, avg_valid_loss,  valid_accuracy)) 

    # Save the model if the reconstruction loss is lower than the best one so far
    if avg_valid_loss < best_loss and epoch < num_epochs:
        torch.save(model.state_dict(), os.path.join(args.output, "{}_{}.model".format(namespace, epoch + 1)))
        best_loss = valid_loss

# Final log
logger.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t) / 60))


