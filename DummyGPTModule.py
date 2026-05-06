import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock


class DummyGPTModule(nn.Module):  
    
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config["vocab_size"], config["embedding_dimension"])
        self.pos_embedding = nn.Embedding(config["context_length"], embedding_dim=config["embedding_dimension"])### Assuming max sequence length of 1000
        self.fc = nn.Linear(config["embedding_dimension"], config["vocab_size"])
        self.final_layer_norm = nn.LayerNorm(config["embedding_dimension"])
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config["num_layers"])
        ])
        self.drop_emb = nn.Dropout(config["dropout_rate"])


    def forward(self, x):
        ###Input will be a matrix of bach_size x sequence_length as no of tokens 
        ##x -> (bacth_size, seq_len)where each value is a token
        batch_size,seq_len = x.shape
        token_embeds = self.tok_embedding(x)  ### (batch_size, seq_len, embedding_dim)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=x.device))
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_layer_norm(x)  
        logits = self.fc(x) ## (batch_size, seq_len, vocab_size)

        return logits
