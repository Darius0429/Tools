import torch
from torch import nn

class ViTEncoder(nn.Module):
    def __init__(self, num_channels: int, output_dim: int, kernel_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # Initialize a simple convolution for patch embedding
        self.encode = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.output_dim,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input tensor must be 4-dimensional (B, C, H, W)"
        assert x.size(1) == self.num_channels, f"Expected input with {self.num_channels} channels, got {x.size(1)}"
        assert x.size(2) % self.kernel_size == 0, f"Height {x.size(2)} must be divisible by kernel size {self.kernel_size}"
        assert x.size(3) % self.kernel_size == 0, f"Width {x.size(3)} must be divisible by kernel size {self.kernel_size}"

        # Apply the convolutional patch embedding
        x = self.encode(x)

        # Reshape to (B, N, D) where N is number of patches and D is output_dim
        x = x.flatten(2).transpose(1, 2)  # (B, D, H', W') -> (B, D, N) -> (B, N, D)
        
        return x
    

