import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from importlib.metadata import version
import tiktoken
# print("tiktoken version:", version("tiktoken"))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# print(device)

context_size = 4
batch_size = 4

# Tokenizes the text
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
# print(len(raw_text))
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
# print(vocab_size)
# print(len(preprocessed))
# print(chars)

#-------------------- TOKENIZER ------------------------ (Encoding and decoding functions for the text)
vocab = {token:integer for integer,token in enumerate(all_words)}

# Tokenizer class:
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    def encode(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int  #A
                        else "<|unk|>" for item in preprocessed]
        encoded_list = [self.str_to_int[s] for s in preprocessed]
        return encoded_list
    def decode(self,encoded_list):
        decoded_text = " ".join([self.int_to_str[i] for i in encoded_list])
        decoded_text = re.sub(r'\s+([,.?!"()\'])', r'\1', decoded_text)
        return decoded_text

#------------------DATASET CLASS------------------------
class GPTDataSetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

#----------------------DATA LOADER-----------------------
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataSetV1(txt,tokenizer,max_length,stride)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
    return dataloader

# testing the dataloader

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=True)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)

#--------------------Embedding layer---------------------
output_dim = 256
vocab_size = 50257
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=True)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = token_embedding_layer(inputs)
# considering position as well
context_length = max_length
pos_embedding_layer = nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings+pos_embeddings
# print(pos_embeddings.shape)
# print(input_embeddings.shape)


#--------------------------------------------------------
