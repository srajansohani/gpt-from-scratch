from torch.utils.data import Dataset
import torch
import tiktoken

class TextDataset(Dataset):
    def __init__(self,tokenizer,text,max_length,stride):
        super().__init__()
        
        self.x = []
        self.y = []

        token_ids = tokenizer.encode(text)
        print(f"Total tokens in text: {len(token_ids)}")

        ##Using sliding window approach to create input and target sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_seq = token_ids[i:i+max_length]
            target_seq = token_ids[i+1:i+1+max_length]
            self.x.append(torch.tensor(input_seq))
            self.y.append(torch.tensor(target_seq))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    


def create_dataloader(text,max_length=256,stride=128,batch_size=4,shuffle=True,drop_last=True,num_workers=0):

    ###Intialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    ###Create dataset and dataloader
    dataset = TextDataset(tokenizer,text,max_length,stride)
    print(dataset.__len__())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,drop_last=drop_last)
    return dataloader