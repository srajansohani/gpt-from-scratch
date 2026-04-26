import torch
import torch.nn as nn



### Feed Forward Layer is a simple MLP that takes the output of the attention layer and processes it further. It consists of two linear layers with a GELU
#  activation in between. The first linear layer expands the dimension of the input, and the second linear layer contracts it back to the original dimension. This allows the model to learn more complex representations of the data after the attention mechanism has been applied.

###Expansion Compression network
class FeedForwardLayer(nn.Module): 

    
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dimension"],4*cfg["embedding_dimension"]),### Expansion
            nn.ReLU(),###Activation
            nn.Linear(4*cfg["embedding_dimension"],cfg["embedding_dimension"])### Contraction back to original dimension
        )

    def forward(self,x):
        return self.layers(x)