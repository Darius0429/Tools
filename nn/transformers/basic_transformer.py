from nn.blocks.transformer import *

class Transformer(nn.Module):
    def __init__(self, dim_tokens:int, num_layers: int, hidden_dim:int, num_heads:int, dropout:float=0.1, normalize_before:bool=True):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(
            c1=dim_tokens,
            cm=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            normalize_before=normalize_before,
        )

        self.layers = nn.ModuleList(
            [self.transformer_layer for _ in range(num_layers)]
        )
        

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x