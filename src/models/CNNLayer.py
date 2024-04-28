import torch
from torch import nn

class CNNLayer(nn.Module):
    """CNN Layer"""

    def __init__(self, input_dim) -> None:
        super().__init__()
        self.Con1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=16, padding='same')
        self.act1 = nn.LeakyReLU()
        self.Con2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16, padding='same')
        self.BN1 = nn.BatchNorm1d(32)
        self.act2 = nn.LeakyReLU()
        self.Pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.Con3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, padding='same')
        self.act3 = nn.LeakyReLU()
        self.Con4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, padding='same')
        self.BN2 = nn.BatchNorm1d(64)
        self.act4 = nn.LeakyReLU()
        self.Pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.Con5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, padding='same')
        self.act5 = nn.LeakyReLU()
        self.Con6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, padding='same')
        self.BN3 = nn.BatchNorm1d(128)
        self.act6 = nn.LeakyReLU()
        self.Pool3 = nn.MaxPool1d(kernel_size=4, stride=4)


    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of CNN.

        Args:
            input_data (torch.Tensor): input data with shape [B, N, D]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.Pool1(self.act2(self.BN1(self.Con2(self.act1(self.Con1(input_data))))))
        hidden = self.Pool2(self.act4(self.BN2(self.Con4(self.act3(self.Con3(hidden))))))
        hidden = self.Pool3(self.act6(self.BN3(self.Con6(self.act5(self.Con5(hidden))))))
        return hidden
