
from torch import nn
from torch.nn import functional as F
import torch

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx,targets=None):
        logits = self.token_embedding_table(idx) #(B,T,C)
        
        if targets is None:
            return logits,None
        
        B,T,C =  logits.shape
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)
        loss2 = F.cross_entropy(logits,targets)

        return logits,loss2
    

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            #Focusting on last step as it is the new prediction
            logits = logits[:,-1,:]

            probs = F.softmax(logits,dim=-1) # (B,C)
            
            idx_next = torch.multinomial(probs,num_samples=1) #(B,1)

            idx = torch.cat((idx,idx_next),dim=1)
        return idx
 

    