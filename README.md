# Convolutional autoencoder for DNA sequences

The project aims at creating a conditional variational autoencoder for DNA sequences, focused on regulatory sequences. Work is still in progress.

To train the model, run:
```
python3 train.py --path INPUT_DIR --filename FILENAME --output OUTPUT_DIR --namespace RUN_NAME
```

To tune hyperparameters with W&B, run:
```
python3 train_new.py --path INPUT_DIR --filename FILENAME --output OUTPUT_DIR --namespace RUN_NAME
```

Other parameters include:
```
  --model NAME          File with the model weights to load before training.
  --dim INT             Dimension of the latent space, default: 64.
  --seed NUMBER         Set random seed, default: 0
  --optimizer NAME      optimization algorithm to use for training the
                        network, default = Adam
  --loss_fn NAME        loss function for training the network, default:
                        MSELoss
  --batch_size INT      size of the batch, default: 64
  --num_workers INT     how many subprocesses to use for data loading,
                        default: 4
  --num_epochs INT      maximum number of epochs to run, default: 300
  --acc_threshold FLOAT
                        threshold of the validation accuracy - if gained
                        training process stops, default: 0.9
  --learning_rate FLOAT
                        initial learning rate, default: 0.001
  --no_adjust_lr        no reduction of learning rate during training,
                        default: False
  --seq_len INT         Length of the input sequences to the network, default:
                        200
  --dropout-conv FLOAT  Dropout of convolutional layers, default value is 0.2
  --dropout-fc FLOAT    Dropout of fully-connected layers, default value is
                        0.5
  --weight-decay FLOAT  Weight decay, default value is 0.0001
  --momentum FLOAT      Momentum, default value is 0.1
  --no_noise            Not replacing Ns in the sequence with random
                        nucleotides, default: False
  --net_type NAME       Network type name, default: CNN
```
