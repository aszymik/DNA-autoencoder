from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def loss_dataframes(path):
    df = pd.read_csv(path, sep='\t', header=0)
    return df[df['Stage'] == 'train'], df[df['Stage'] == 'valid']


def adjust_axes(ax, y_bottom=None, y_top=None):
    # Adjust plot parameters
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_facecolor('#f1f1f1')
    ax.grid(color='white', alpha=0.5)
    ax.set_axisbelow(True)
    if (y_bottom is not None) and (y_top is not None):
        ax.set_ylim(y_bottom, y_top)


def plot_loss(path, title, latent_dim):
    df_train, df_valid = loss_dataframes(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x = df_train['Epoch'], y = df_train[f'Loss'], s=3, alpha=0.6, label='training')
    ax.scatter(x = df_valid['Epoch'], y = df_valid[f'Loss'], s=3, alpha=0.6, label='validation')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    adjust_axes(ax, 1.03, 1.38)

    fig.suptitle(title, fontsize=15)
    plt.title(f'Latent dim = {latent_dim}', fontsize=13)
    plt.legend()
    plt.show()

def compare_loss(paths, title, latent_dims):
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, latent_dim in zip(paths, latent_dims):
        _, df_valid = loss_dataframes(path)
        ax.scatter(x = df_valid['Epoch'], y = df_valid[f'Loss'], s=3, alpha=0.6, label=f'latent dim {latent_dim}')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    adjust_axes(ax, 1.03, 1.38)

    fig.suptitle(title, fontsize=15)
    plt.title('Validation loss comparison')
    plt.legend()
    plt.show()

def compare_losses(paths, title, latent_dims):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for path, latent_dim in zip(paths, latent_dims):
        df_train, df_valid = loss_dataframes(path)
        axes[0].scatter(x = df_train['Epoch'], y = df_train[f'Loss'], s=3, alpha=0.6, label=f'latent dim {latent_dim}')
        axes[1].scatter(x = df_valid['Epoch'], y = df_valid[f'Loss'], s=3, alpha=0.6, label=f'latent dim {latent_dim}')

    adjust_axes(axes[0], 0.93, 1.38)
    adjust_axes(axes[1], 0.93, 1.38)
    axes[0].set_title('Training')
    axes[1].set_title('Validation')

    fig.supxlabel('Epoch', fontsize=13)
    fig.supylabel('Loss', fontsize=13)
    
    fig.suptitle(title, fontsize=15)
    plt.legend(fontsize=9.5)
    plt.show()    



dim_64_2022_200bp = 'data/2022_200bp/first_test_train_results.tsv'
dim_100_2022_200bp = 'data/2022_200bp/dim_100_first_test_train_results.tsv'
dim_200_2022_200bp = 'data/2022_200bp/dim_200_first_test_train_results.tsv'
dim_400_2022_200bp = 'data/2022_200bp/dim_400_first_test_train_results.tsv'

dim_100_lr_0001 = 'data/2022_200bp/dim_100_lr_0001_train_results.tsv'
dim_200_lr_0001 = 'data/2022_200bp/dim_200_lr_0001_train_results.tsv'
dim_400_lr_0001 = 'data/2022_200bp/dim_400_lr_0001_train_results.tsv'

if __name__ == '__main__':
    # plot_loss(dim_64_2022_200bp, 'Training autoencoder on 200 bp 2022 data (promoter active)', latent_dim=64)
    # plot_loss(dim_100_2022_200bp, 'Training autoencoder on 200 bp 2022 data (promoter active)', latent_dim=100)
    compare_losses([dim_64_2022_200bp, dim_100_2022_200bp, dim_200_2022_200bp, dim_400_2022_200bp], 
                 'Training autoencoder on 200 bp 2022 data (promoter active)', 
                 [64, 100, 200, 400])
    # compare_losses([dim_100_lr_0001, dim_200_lr_0001, dim_400_lr_0001], 
    #              'Training autoencoder on 200 bp 2022 data (promoter active)', 
    #              [100, 200, 400])