import torch
# from CasualAttention import CausualAttentionLayer
# from MultiHeadAttention import MultiHeadAttentionLayer
from LayerNormalization import LayerNormalization
from TransformerBlock import TransformerBlock
from DummyGPTModule import DummyGPTModule
import tiktoken

# text = "your journey starts with one step"
config = {
    "embedding_dimension":768,
    "context_length":1024,
    "num_heads":2,
    "dropout_rate":0.1,
    "qkv_bias": False,
    "vocab_size": 50257,
    "num_layers":6,
} 





# casual_attention_layer = CausualAttentionLayer(3,3)

# output = casual_attention_layer(inputs)

# print(output)

# inputs = torch.stack([inputs,inputs]) ## here we are stacking the same input 3 times to create a batch of 3 sequences of 6 tokens each with 3 dimensional embeddings

# print(inputs.shape)

# multi_head_attention_layer = MultiHeadAttentionLayer(3,3,6,2) 
model = DummyGPTModule(config)






# z = multi_head_attention_layer(inputs)
# z = gpt(inputs)

# print(torch.argmax(z,dim=-1))

# print(z.shape,z)

def text_to_token_id(text,tokenizer):
    tokens = tokenizer.encode(text)
    encoded_tensor = torch.tensor(tokens).unsqueeze(0)  ## Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0).tolist()  ## Remove batch dimension and convert to list
    text = tokenizer.decode(flat)
    return text

def generate_text_simple(model, input_tokens, max_length,contex_size):
    for _ in range(max_length):

        input_under_context = input_tokens[:,-contex_size:]
        with torch.no_grad():
              logits = model(input_under_context)
       
        logits = logits[:,-1,:]  ## (batch_size, tokens, vocab_size)

        probabs = torch.softmax(logits,dim=-1) ## (batch_size, vocab_size)

        next_token = torch.argmax(probabs, dim=-1,keepdim=True)

        input_tokens = torch.cat((input_tokens, next_token), dim=1)
    return input_tokens


text = "Hello I am"

tokenizer = tiktoken.encoding_for_model("gpt2")

input_tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0) 

generated_tokens = generate_text_simple(model, input_tokens, max_length=10, contex_size=config["context_length"])
print(tokenizer.decode(generated_tokens[0].tolist())) 


