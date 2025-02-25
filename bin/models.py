import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# sprawdzic wartosci bezwgledne czy przewiduje same zera
# jeśli tak, to inny optimizer
# jeśli nie, to inna architektura
# przy stracie mozna brac maksimum i tylko do tego

# cross-entropy loss zamiast mse

# zwiekszyc liczbe przykladow
# zwiekszyc liczbe filtrow w encoderze / przejsc do sieci fc


# potem: zobaczyc ile filtrow jest wyzerowanych

# poł na pół: encoder cnn i decoder fc

# TODO: zobacz jaki jest output encoder conv i dostosuj kształty warstw fc

class CNNAutoencoder(nn.Module):
    # num_channels wcześniej: [32, 64, 128]
    def __init__(self, seq_len=200, latent_dim=200, num_channels=[256, 128, 64], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4], 
                 out_paddings=[0, 2, 1], fc_layers=[256], fc_dropout=0.3, conv_dropout=0.15):    
        super(CNNAutoencoder, self).__init__()

        num_channels = [1] + num_channels
        self.num_channels = num_channels
        self.seq_len = seq_len
        paddings = [int((w-1)/2) for w in kernel_widths]

        # Track compressed sequence length after pooling in the encoder
        compressed_seq_len = seq_len
        for pooling in pooling_widths:
            compressed_seq_len = math.ceil(compressed_seq_len / pooling)
        self.compressed_seq_len = compressed_seq_len

        # Encoder: Conv2d, BatchNorm2d, ReLU, MaxPool2d
        conv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling) in enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1  # 4 for the first layer
            conv_modules += [
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=conv_dropout),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
        self.conv_layers = nn.Sequential(*conv_modules)

        # Fully connected layers for latent space
        self.fc_input = self.compressed_seq_len * num_channels[-1]

        encoder_fc_modules = []
        fc_layers = [self.fc_input] + fc_layers + [latent_dim]
        for in_shape, out_shape in zip(fc_layers[:-1], fc_layers[1:]):
            encoder_fc_modules += [
                nn.Linear(in_shape, out_shape),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=fc_dropout)
            ]
        self.encoder_fc = nn.Sequential(*encoder_fc_modules)

        # Decoder: Fully connected and ConvTranspose2d to recover original dimensions
        decoder_fc_modules = []
        for in_shape, out_shape in zip(reversed(fc_layers[1:]), reversed(fc_layers[:-1])):
            decoder_fc_modules += [
                nn.Linear(in_shape, out_shape),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=fc_dropout)
            ]
        self.decoder_fc = nn.Sequential(*decoder_fc_modules)

        deconv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling, out_pad) in enumerate(zip(reversed(num_channels[1:]), reversed(num_channels[:-1]), reversed(kernel_widths), 
                                                                            reversed(paddings), reversed(pooling_widths), out_paddings)):
            if num < (len(num_channels)-2):
                deconv_modules += [
                    nn.ConvTranspose2d(in_channels=in_ch, 
                                       out_channels=out_ch, 
                                       kernel_size=(1, kernel), 
                                       stride=(1, pooling), 
                                       padding=(0, padding),
                                       output_padding=(0, out_pad)),
                    nn.BatchNorm2d(out_ch),
                    # nn.ReLU(),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(p=conv_dropout)
                ]
            else:
                deconv_modules += [
                    nn.ConvTranspose2d(in_channels=in_ch, 
                                       out_channels=out_ch, 
                                       kernel_size=(4, kernel), 
                                       stride=(1, pooling), 
                                       padding=(0, padding),
                                       output_padding=(0, out_pad)),
                ]
        self.deconv_layers = nn.Sequential(*deconv_modules)
        self.output_activation = nn.Softmax(dim=2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder forward pass
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)

        # Decoder forward pass
        x = self.decoder_fc(x)
        x = x.view(x.size(0), -1, 1, self.compressed_seq_len)  # Unflatten for deconv layers
        x = self.deconv_layers(x)
        # return x  # crossentropyloss ma wbudowane logSoftmax
        return self.output_activation(x)
    
# CNN VAE
    
class Encoder(nn.Module):
    def __init__(self, latent_dim, num_channels, kernel_widths, pooling_widths, paddings,
                 fc_layers, fc_dropout, conv_dropout, fc_input):    
        super(Encoder, self).__init__()

        # Conv2d, BatchNorm2d, ReLU, MaxPool2d
        conv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling) in enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1  # 4 for the first layer
            conv_modules += [
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=conv_dropout),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
        self.conv_layers = nn.Sequential(*conv_modules)

        # FC layers
        encoder_fc_modules = []
        fc_layers = [fc_input] + fc_layers
        for in_shape, out_shape in zip(fc_layers[:-1], fc_layers[1:]):
            encoder_fc_modules += [
                nn.Linear(in_shape, out_shape),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=fc_dropout)
            ]
        self.encoder_fc = nn.Sequential(*encoder_fc_modules)

        # Latent space
        self.mu_layer = nn.Linear(fc_layers[-1], latent_dim)
        self.logvar_layer = nn.Linear(fc_layers[-1], latent_dim)  # log for stability and not constraining to positive values
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, num_channels, kernel_widths, pooling_widths, paddings,
                 out_paddings, fc_layers, fc_dropout, conv_dropout, compressed_seq_len, fc_input):    
        super(Decoder, self).__init__()

        self.compressed_seq_len = compressed_seq_len

        # FC and ConvTranspose2d to recover original dimensions
        fc_layers = [fc_input] + fc_layers + [latent_dim]
        decoder_fc_modules = []
        for in_shape, out_shape in zip(reversed(fc_layers[1:]), reversed(fc_layers[:-1])):
            decoder_fc_modules += [
                nn.Linear(in_shape, out_shape),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=fc_dropout)
            ]
        self.decoder_fc = nn.Sequential(*decoder_fc_modules)

        deconv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling, out_pad) in enumerate(zip(reversed(num_channels[1:]), reversed(num_channels[:-1]), reversed(kernel_widths), 
                                                                            reversed(paddings), reversed(pooling_widths), out_paddings)):
            if num < (len(num_channels)-2):
                deconv_modules += [
                    nn.ConvTranspose2d(in_channels=in_ch, 
                                       out_channels=out_ch, 
                                       kernel_size=(1, kernel), 
                                       stride=(1, pooling), 
                                       padding=(0, padding),
                                       output_padding=(0, out_pad)),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(p=conv_dropout)
                ]
            else:
                deconv_modules += [
                    nn.ConvTranspose2d(in_channels=in_ch, 
                                       out_channels=out_ch, 
                                       kernel_size=(4, kernel), 
                                       stride=(1, pooling), 
                                       padding=(0, padding),
                                       output_padding=(0, out_pad)),
                ]
        self.deconv_layers = nn.Sequential(*deconv_modules)
        self.output_activation = nn.Softmax(dim=2)

    def forward(self, z):
        x = self.decoder_fc(x)
        x = x.view(x.size(0), -1, 1, self.compressed_seq_len)  # Unflatten for deconv layers
        x = self.deconv_layers(x)
        return self.output_activation(x)


class VAE(nn.Module):
    def __init__(self, seq_len=200, latent_dim=200, num_channels=[256, 128, 64], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4], 
                 out_paddings=[0, 2, 1], fc_layers=[256], fc_dropout=0.3, conv_dropout=0.15):    
        super(VAE, self).__init__()

        self.num_channels = [1] + num_channels
        self.paddings = [int((w-1)/2) for w in kernel_widths]

        # Track compressed sequence length after pooling in the encoder
        compressed_seq_len = seq_len
        for pooling in pooling_widths:
            compressed_seq_len = math.ceil(compressed_seq_len / pooling)
        self.compressed_seq_len = compressed_seq_len

        self.fc_input = compressed_seq_len * num_channels[-1]
        
        # Encoder and decoder
        self.encoder = Encoder(latent_dim, self.num_channels, kernel_widths, pooling_widths, self.paddings, fc_layers, fc_dropout, conv_dropout, self.fc_input)
        self.decoder = Decoder(latent_dim, self.num_channels, kernel_widths, pooling_widths, self.paddings, out_paddings, fc_layers, fc_dropout, conv_dropout, self.compressed_seq_len, self.fc_input)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # convert log variance to standard deviation
        eps = torch.randn_like(std)  # sample from N(0, I)
        return eps * std + mu

    def sample_latent(self, x):
        mu, logvar = self.encoder(x)
        return self.reparametrize(mu, logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def KL_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def ELBO_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = KL_divergence(mu, logvar)
    return recon_loss + kl_loss


class SemiCNNAutoencoder(nn.Module):
    def __init__(self, seq_len=200, latent_dim=64, num_channels=[256, 128, 64], kernel_widths=[19, 11, 7], 
                 pooling_widths=[3, 4, 4], decoder_sizes=[256, 512, 512, 1024], dropout=0.15):    
        super(SemiCNNAutoencoder, self).__init__()

        num_channels = [1] + num_channels
        self.num_channels = num_channels
        self.seq_len = seq_len
        paddings = [int((w-1)/2) for w in kernel_widths]

        # Track compressed sequence length after pooling in the encoder
        compressed_seq_len = seq_len
        for pooling in pooling_widths:
            compressed_seq_len = math.ceil(compressed_seq_len / pooling)
        self.compressed_seq_len = compressed_seq_len

        # Encoder: Conv2d, BatchNorm2d, ReLU, MaxPool2d
        conv_modules = []
        for num, (in_ch, out_ch, kernel, padding, pooling) in enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1  # 4 for the first layer
            conv_modules += [
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=dropout),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
        self.conv_layers = nn.Sequential(*conv_modules)

        # Fully connected layers for latent space
        self.fc_input = self.compressed_seq_len * num_channels[-1]
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.fc_input, latent_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout)
        )

        # Decoder: Fully connected layers
        decoder_layers = [
            nn.Linear(latent_dim, self.fc_input),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout),
        ]
        decoder_sizes.insert(0, self.fc_input)

        for input, output in zip(decoder_sizes[:-1], decoder_sizes[1:]):
            decoder_layers.append(nn.Linear(input, output))
            decoder_layers.append(nn.LeakyReLU(negative_slope=0.01))
            decoder_layers.append(nn.Dropout(p=dropout*2))
        
        decoder_layers.append(nn.Linear(decoder_sizes[-1], 4 * seq_len))
        self.decoder_fc = nn.Sequential(*decoder_layers)

        self.output_activation = nn.Softmax(dim=2)  # Softmax for 4x200 output

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder forward pass
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)

        # Decoder forward pass
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 1, 4, self.seq_len)  # Reshape to match original sequence dimensions
        return self.output_activation(x)