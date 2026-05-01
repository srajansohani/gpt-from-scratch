# As multihead attention is just multiple attention heads working in parallel, 
# we can implement it by creating multiple sets of Wq, Wk, Wv matrices and then 
# concatenating the outputs of each head together. Below is an example implementation o
# f multihead attention in PyTorch.

import torch
import torch.nn as nn
class MaskedMultiHeadAttentionLayer(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,droput=0.2):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.dropout = nn.Dropout(droput)
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_in, d_out * num_heads) ## We multiplied with num_heads as mentioned in notes we need to create multiple sets of Wq, Wk, Wv matrices for each head. So we can do this by creating a single linear layer that outputs d_out * num_heads dimensions, and then we can split this output into num_heads separate heads later on in the forward pass.
        self.Wk = nn.Linear(d_in, d_out * num_heads)
        self.Wv = nn.Linear(d_in, d_out * num_heads)
        self.out_proj = nn.Linear(d_out * num_heads, d_in) ## This is the final linear layer that will project the concatenated output of all the heads back to the original input dimension d_in, so that we can add it back to the input (residual connection) and pass it through a feedforward network in the transformer architecture.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), ###Inout tokes are called the context length
                       diagonal=1)
        )## this will create a lower triangular matrix of ones with the shape (1, 1, context_length, context_length), which will be used to mask out the future tokens in the attention scores
    
    def forward(self,x):
        b,num_tokens, d_in = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        ###unrolling the last dimension d_out * num_heads into two dimensions num_heads and d_out, so that we can perform the attention calculation for each head separately.
        Q = Q.view(b,num_tokens,self.num_heads,self.d_out) 
        K = K.view(b,num_tokens,self.num_heads,self.d_out)
        V = V.view(b,num_tokens,self.num_heads,self.d_out)

        ### Reshaping the tensors to bring the num_heads dimension before the num_tokens dimension, so that we can perform the attention calculation for each head separately.
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        ## Computed the scaled dot product attention (aka self attention) for each head separately. This involves taking the dot product of the query and key matrices, scaling it by the square root of the dimension of the key vectors, and then applying a softmax function to obtain the attention weights.
        attention_scores = Q @ K.transpose(2,3)   ###(Dimesion become ()b,num_heads,num_tokens,num_tokens) as we are taking transpose of K matrix to make it compatible for matrix multiplication with Q matrix

        
        ###Masking the attention scores to prevent the model from attending to future tokens in the sequence. This is done by creating a mask that has a value of -inf for positions that correspond to future tokens, and then adding this mask to the attention scores before applying the softmax function.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] ## this will create a boolean mask of shape (num_tokens, num_tokens) where the upper triangular part (corresponding to future tokens) is True and the lower triangular part (corresponding to past tokens) is False

        attention_scores.masked_fill(mask_bool,-torch.inf)


        attention_weights = torch.softmax(attention_scores / (self.d_out ** 0.5), dim=-1) ## this will apply the softmax function to the masked attention scores, giving us the attention weights that sum to 1 for each token in the sequence
        attention_weights = self.dropout(attention_weights)  ### For gradient clipping, we can add a dropout layer

        ## Finally, we compute the output of the multi-head attention layer by taking the weighted sum of the value vectors, where the weights are given by the attention weights. This will give us a tensor of shape (b, num_heads, num_tokens, d_out) which we can then reshape back to (b, num_tokens, d_out * num_heads) to get the final output of the multi-head attention layer.

        out = attention_weights @ V ## this will give us a tensor of shape (b, num_heads, num_tokens, d_out) which is the output of the multi-head attention layer for each head separately

        out = out.transpose(1,2).contiguous().view(b,num_tokens,self.d_out * self.num_heads) ## this will reshape the output tensor back to (b, num_tokens, d_out * num_heads) by first transposing the num_heads and num_tokens dimensions, and then using the view function to combine the num_heads and d_out dimensions into a single dimension of size d_out * num_heads

        ## we have got the z(output) for each head now we have to calculate the final context vector

        context_vector = self.out_proj(out) ## this will project the concatenated output of all the heads back to the original input dimension d_in, so that we can add it back to the input (residual connection) and pass it through a feedforward network in the transformer architecture.
        return context_vector

