from torch import nn


def autoencoder_nn(input_dim: int, latent_dim: int) -> nn.Sequential:
    '''
    Autoencoder network.
    :param input_dim: Input dimensions (features  count).
    :param latent_dim: Bottleneck dimension.
    :return:
    '''
    model = nn.Sequential(
        # Encode
        nn.Linear(input_dim, 1024),
        nn.BatchNorm1d(1024),  # batch normalize
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, latent_dim),
        nn.ReLU(),
        # Decode
        nn.Linear(latent_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, input_dim),
    )
    return model