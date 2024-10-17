import torch.nn as nn
import math


class CNNAutoencoder(nn.Module):
    def __init__(self, seq_len=200, latent_dim=64, num_channels=[32, 64, 128], kernel_widths=[5, 5, 5], paddings=[2, 2, 2], pooling_widths=[2, 2, 2]):
        super(CNNAutoencoder, self).__init__()

        num_channels = [1] + num_channels
        self.num_channels = num_channels
        self.seq_len = seq_len

        # Track compressed sequence length after pooling
        compressed_seq_len = seq_len
        for pooling in pooling_widths:
            compressed_seq_len = math.ceil(compressed_seq_len / pooling)
        print(f'compressed seq len: {compressed_seq_len}')
        self.compressed_seq_len = compressed_seq_len  # Final compressed length after encoder


        # Encoder: Conv2d, BatchNorm2d, ReLU, MaxPool2d
        conv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling) in enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1  # 4 for first layer to match your example
            conv_modules += [[
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]]  # potem usunąć podwójną listę
        
        self.conv1 = nn.Sequential(*conv_modules[0])
        self.conv2 = nn.Sequential(*conv_modules[1])
        self.conv3 = nn.Sequential(*conv_modules[2])

        # self.conv_layers = nn.Sequential(*conv_modules)

        # Fully connected layers for latent space
        self.fc_input = self.compressed_seq_len * num_channels[-1]
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.fc_input, latent_dim),
            nn.ReLU()
        )

        # Decoder: Fully connected and ConvTranspose2d to recover original dimensions
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.fc_input),
            nn.ReLU()
        )

        deconv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling) in enumerate(zip(reversed(num_channels[1:]), reversed(num_channels[:-1]), reversed(kernel_widths), reversed(paddings), reversed(pooling_widths))):
            k = 4 if num == (len(num_channels)-2) else 1
            print(f'input: {in_ch}')
            print(f'output: {out_ch}')
            deconv_modules += [[
                nn.ConvTranspose2d(in_channels=in_ch, 
                                   out_channels=out_ch, 
                                   kernel_size=(k, kernel), 
                                   stride=(1, pooling), 
                                   # padding=(0, padding),
                                   output_padding=(0, pooling - 1) if pooling > 1 else 0),
                                   # output_padding=(0, padding))
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ]] # tak samo usunąć podwójną listę
        
        self.deconv1 = nn.Sequential(*deconv_modules[0])
        self.deconv2 = nn.Sequential(*deconv_modules[1])
        self.deconv3 = nn.Sequential(*deconv_modules[2])
        # self.deconv_layers = nn.Sequential(*deconv_modules)

        # Final layer for output
        self.output_activation = nn.Softmax(dim=1)  # Softmax over the channels

    def forward(self, x):
        # Encoder forward pass
        print(f'at the beginning: {x.shape}')
        # x = self.conv_layers(x)
 
        x = self.conv1(x)
        print(f'after conv 1: {x.shape}')
        x = self.conv2(x)
        print(f'after conv 2: {x.shape}')
        x = self.conv3(x)
        print(f'after conv 3: {x.shape}')


        x = x.view(x.size(0), -1)  # Flatten
        print(f'before encoder fc: {x.shape}')

        x = self.encoder_fc(x)
        print(f'after encoder fc: {x.shape}')

        # Decoder forward pass
        x = self.decoder_fc(x)
        print(f'after decoder fc: {x.shape}')

        x = x.view(x.size(0), -1, 1, self.compressed_seq_len)  # Unflatten for deconv layers
        print(f'before deconv: {x.shape}')
        # x = self.deconv_layers(x)

        x = self.deconv1(x)
        print(f'after deconv 1: {x.shape}')
        x = self.deconv2(x)
        print(f'after deconv 2: {x.shape}')
        x = self.deconv3(x)
        print(f'after deconv 3: {x.shape}')
        return self.output_activation(x)  


class CNN1DAutoencoder(nn.Module):
    def __init__(self, seq_len=200, input_channels=4, latent_dim=64):
        super(CNN1DAutoencoder, self).__init__()

        # zmienić na 2D, kernel 4

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding=2),  # (batch_size, 32, 200)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (batch_size, 32, 100)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, 50)
            nn.ReLU(),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, 25)
            nn.ReLU(),
            
            nn.Flatten(),  # (batch_size, 128 * 25)
            nn.Linear(in_features=128 * 25, out_features=latent_dim),  # (batch_size, latent_dim)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128 * 25),  # (batch_size, 128 * 25)
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 25)),  # (batch_size, 128, 25)
            
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),  # (batch_size, 64, 50)
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),  # (batch_size, 32, 100)
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # (batch_size, 4, 200)
            nn.Softmax(dim=1)  # Apply softmax along the channel dimension (A, C, G, T)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    