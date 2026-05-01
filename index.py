import torch
# from CasualAttention import CausualAttentionLayer
# from MultiHeadAttention import MultiHeadAttentionLayer
from LayerNormalization import LayerNormalization
from TransformerBlock import TransformerBlock

# text = "your journey starts with one step"
config = {
    "embedding_dimension":3,
    "context_length":1024,
    "num_heads":2,
    "dropout_rate":0.1,
    "qkv_bias": False
} 
inputs = torch.tensor([
    [0.43,0.15,0.89], #your   (x^1)
    [0.55,0.87,0.66], #journey (x^2)
    [0.57,0.85,0.64], #starts  (x^3)
    [0.22,0.58,0.33], #with    ((x^4)
    [0.77,0.25,0.10], #one     ()
    [0.05,0.80,0.55]# step
])

# casual_attention_layer = CausualAttentionLayer(3,3)

# output = casual_attention_layer(inputs)

# print(output)

inputs = torch.stack([inputs,inputs]) ## here we are stacking the same input 3 times to create a batch of 3 sequences of 6 tokens each with 3 dimensional embeddings

# print(inputs.shape)

# multi_head_attention_layer = MultiHeadAttentionLayer(3,3,6,2) 
transformer_block = TransformerBlock({
    "embedding_dimension":3,
    "context_length":6,
    "num_heads":2,
    "dropout_rate":0.1
})




# z = multi_head_attention_layer(inputs)
z = transformer_block(inputs)
print(z.shape,z)