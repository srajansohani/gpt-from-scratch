import torch
# from CasualAttention import CausualAttentionLayer
# from MultiHeadAttention import MultiHeadAttentionLayer
from LayerNormalization import LayerNormalization
from TransformerBlock import TransformerBlock
from DummyGPTModule import DummyGPTModule
import tiktoken

config = {
    "embedding_dimension":768,
    "context_length":1024,
    "num_heads":2,
    "dropout_rate":0.1,
    "qkv_bias": False,
    "vocab_size": 50257,
    "num_layers":6,
} 


with open('mini-shakespear.txt','r') as file:
    content = file.read()



all_tokens = tiktoken.encoding_for_model("gpt2").encode(content)



def generate_blocks(tokens,block_size):
    blocks = []
    for i in range(0,len(tokens)-block_size,block_size):
        block = tokens[i:i+block_size]
        blocks.append(block)
    return blocks

blocks = generate_blocks(all_tokens,block_size=10)

print(blocks)