import os
import sys
import re
import torch
import tiktoken 
with open('mini-shakespear.txt','r',encoding='utf-8') as f:
    text = f.read()

# with open('chat.txt','r',encoding='utf-8') as c:
#     lines = c.readlines()



# cleaned_lines = []

# pattern = r"^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} - "

# for line in lines:
#     cleaned = re.sub(pattern, "", line)
#     cleaned_lines.append(cleaned)

# for line in cleaned_lines:
#     print(line)




##Tokeniztion based on chars


chars = list(set(text))

## stoi = {} , for i,ch enumerate(chars): stoi.add({ch: i})
stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars) }

#Encoder generates 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# tokenizer = tiktoken.get_encoding('gpt2')
# encode = lambda s: tokenizer.encode(s)
# decode = lambda l: tokenizer.decode(l)


data = torch.tensor(encode(text),dtype = torch.long)
print(data,data.shape)

n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]


block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]

def printExample(x,y):
    for t in range(block_size):
        context = x[: t + 1]
        target = y[t]
        print(f"when input is: {context} then target is: {target}")


torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    print(ix)
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    
    return x,y

xb,yb = get_batch('train')
print(xb.shape,xb,yb.shape,yb)