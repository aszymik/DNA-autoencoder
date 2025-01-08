# Convolutional autoencoder for DNA sequences

The project aims at creating a conditional variational autoencoder for DNA sequences, focused on regulatory sequences. Work is still in progress.

To train the model, run ```python3 train.py --path INPUT_DIR --filename FILENAME --output OUTPUT_DIR --namespace RUN_NAME```.

To tune hyperparameters with W&B, run ```python3 train_new.py --path INPUT_DIR --filename FILENAME --output OUTPUT_DIR --namespace RUN_NAME```.
