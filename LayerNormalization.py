import torch
import torch.nn as nn


### LayerNormalization is used to prevent explodin/vanishing gradient problems as gradient depends on the output of previous layer
class LayerNormalization(nn.Module):

    def __init__(self,embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.scale = nn.Parameter(torch.ones(embedding_dim))  ## this is a learnable parameter that allows the model to scale the normalized output, which can help the model to learn more complex representations and improve its performance. By initializing it to ones, we ensure that the initial output of the layer normalization is not scaled, allowing the model to learn the appropriate scaling during training.
        self.shift = nn.Parameter(torch.zeros(embedding_dim)) ## this is a learnable parameter that allows the model to shift the normalized output, which can help the model to learn more complex representations and improve its performance. By initializing it to zeros, we ensure that the initial output of the layer normalization is not shifted, allowing the model to learn the appropriate shifting during training.
        self.eps = 1e-5  ### this is a small constant added to the variance to prevent division by zero when normalizing the input. It ensures numerical stability during training, especially when the variance is very small, which can occur when the input features have low variability. By adding this small value, we avoid potential issues with infinite or undefined values in the normalized output.
    
      
    def forward(self,x):
        
        mean = x.mean(dim=-1,keepdim=True) 
        var = x.var(dim=-1,keepdim=True,unbiased=False)

        ## the mean and variance are calculated along the last dimension of the input tensor x, which typically corresponds to the feature dimension. The keepdim=True argument ensures that the output tensors for mean and variance have the same number of dimensions as the input tensor, allowing for proper broadcasting during normalization.
        ### the formula for layer normalization is: 

        x_normalize = (x - mean)/torch.sqrt(var + self.eps)


        out = self.scale * x_normalize + self.shift

        return out