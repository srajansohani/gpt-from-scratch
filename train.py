
import tiktoken
import torch
import re
from utils import calc_loss_loader,calculate_batch_loss,generate_text_simple,evaluate_model
from DummyGPTModule import DummyGPTModule
import time

from TextDataClass import create_dataloader

def remove_consecutive_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()
config = {
    "embedding_dimension":768,
    "context_length":256,
    "num_heads":2,
    "dropout_rate":0.1,
    "qkv_bias": False,
    "vocab_size": 50257,
    "num_layers":6,
} 

##training params
##Context_Size , Stride
context_size = 256
stride = 128


tokenizer = tiktoken.encoding_for_model("gpt2")
with open ('data.txt','r') as file:
    content = file.read()


pattern = r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2} - "
content = re.sub(pattern, "", content)

model = DummyGPTModule(config=config)



train_ratio = 0.9
test_ratio = 0.1
split_idx = int(len(content) * train_ratio)
train_data = content[:split_idx]
test_data = content[split_idx:]


train_loader = create_dataloader(text=train_data,max_length=context_size,stride=stride,batch_size=2,shuffle=True,drop_last=True,num_workers=0)

test_loader = create_dataloader(text=test_data,max_length=context_size,stride=stride,batch_size=2,shuffle=False,drop_last=True,num_workers=0)


# print(len(train_loader),len(test_loader))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004,weight_decay=0.1)

start_time = time.time()
### Training loop

train_losses,val_losses,track_tokens_seen = [],[],[]
epochs = 10
global_step,tokens_seen = 0,-1
eval_freq = 20

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0
    for i, (batch_input, batch_target) in enumerate(train_loader):
        optimizer.zero_grad()
        if i >= len(train_loader):
            break
        batch_input = batch_input
        batch_target = batch_target
        loss = calculate_batch_loss(batch_input, batch_target, model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        tokens_seen += batch_input.numel()
        global_step += 1


        ###optional evaluation step 
       
        if global_step % eval_freq == 0:
            train_loss,val_loss = evaluate_model(model,train_loader,test_loader,eval_tier=eval_freq)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                  f"Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Tokens seen {tokens_seen}"
                  )

    ##Print a sample text after each epoch
    print("Performance check After each epoch: ", generate_text_simple(model, "Money's only excuse", max_length=6, contex_size=context_size))
   


end_time = time.time() 

print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")




    



