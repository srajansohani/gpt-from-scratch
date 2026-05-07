import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def calculate_batch_loss(input_batch,target_batch,model):
    logits = model(input_batch)  ## (batch_size, seq_len, vocab_size)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader,model,num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (batch_input, batch_target) in enumerate(data_loader):
        if i >= num_batches:
            break
        batch_input = batch_input
        batch_target = batch_target
        loss = calculate_batch_loss(batch_input, batch_target, model)
        total_loss += loss.item()

    return total_loss / num_batches


def generate_text_simple(model, input_text, max_length,contex_size):

    input_tokens = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0) 
    for _ in range(max_length):

        input_under_context = input_tokens[:,-contex_size:]
        with torch.no_grad():
              logits = model(input_under_context)
       
        logits = logits[:,-1,:]  ## (batch_size, tokens, vocab_size)

        probabs = torch.softmax(logits,dim=-1) ## (batch_size, vocab_size)

        next_token = torch.argmax(probabs, dim=-1,keepdim=True)

        input_tokens = torch.cat((input_tokens, next_token), dim=1)
    print(input_tokens)
    return tokenizer.decode(input_tokens[0].tolist())


    
def evaluate_model(model,train_loader,val_loader,eval_tier):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_tier)
        val_loss = calc_loss_loader(val_loader, model, num_batches=eval_tier)
    model.train()
    return train_loss,val_loss