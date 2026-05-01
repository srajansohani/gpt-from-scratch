### As we have all the components now we only need to code the architecture
import torch.nn as nn
from LayerNormalization import LayerNormalization
from MaskedMultiHeadAttentionLayer import MaskedMultiHeadAttentionLayer
from FeedForwardLayer import FeedForwardLayer

##Note it is important to have same dimension of inpput and output for each layer. If not, you will
class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(config["embedding_dimension"])
        self.masked_attention = MaskedMultiHeadAttentionLayer(
            d_in = config["embedding_dimension"],
            d_out = config["embedding_dimension"],
            context_length= config["context_length"],
            num_heads= config["num_heads"],
            droput=config["dropout_rate"],
            )
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.ff = FeedForwardLayer(config)
        self.layer_norm_2 = LayerNormalization(config["embedding_dimension"])



    def forward(self,x):
       
        shortcut = x ## This is the residual connection, we will add this back to the output of the attention layer and the feedforward layer later on in the forward pass.
        ## First we will apply layer normalization to the input, then we will pass it through the masked multi-head attention layer, and then we will add the output of the attention layer back to the input (residual connection) and apply dropout. After that, we will pass the result through the feedforward layer, and then add the output of the feedforward layer back to the result of the previous step (residual connection) and apply layer normalization again.
        x_norm = self.layer_norm_1(x)
        attention_out = self.masked_attention(x_norm)
        x = x + self.dropout(attention_out)

        x = x + shortcut

        shortcut = x

        x = self.layer_norm_2(x)
        x = self.ff(x)
        x = self.dropout(x)

        x = x + shortcut

        return x
