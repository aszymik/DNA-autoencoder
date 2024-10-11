import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, input_length=200, input_channels=4, latent_dim=64):
        super(CNNAutoencoder, self).__init__()

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
    