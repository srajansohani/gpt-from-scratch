import torch
import torch.nn as nn


class DummyGPTModle(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
        self.fc = nn.Linear(50, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        print(x.shape)
        output = self.fc(x)
        return output