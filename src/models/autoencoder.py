from torch import nn


def autoencoder_nn(input_dim):
    model = nn.Sequential(
        # Encode
        nn.Linear(input_dim, 1024),
        nn.BatchNorm1d(1024),  # batch normalize
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        # Decode
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, input_dim),
    )
    return model