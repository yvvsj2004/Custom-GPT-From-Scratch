import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

block_size = 8
batch_size = 4

with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
# print(chars)

#-------------------- TOKENIZER ------------------------ (Encoding and decoding functions for the text)
string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long) # Create a tensor for the encoded text

#-------------------------------------------------------
# Spilt the entire text into training set and testing set
n = int(0.8*len(data))
train_data = data[:n]
test_data = data[n:]
#-------------------------------------------------------
x = train_data[:block_size]
y = train_data[1:block_size+1]
for i in range(block_size):
    inpooo = x[:i+1]
    outpoo = y[i]
    # print(f"When input is {inpooo}, output is {outpoo}")
# Block size is the length of the sequence and batch size is how many of these are we doing at the same time
# --------------------get_batch-------------------------
def get_batch(split):
    data = train_data if split == 'train' else test_data
    tensorRand = torch.randint(0, len(data)-block_size, (batch_size,))
    # print(tensorRand)

    x = torch.stack([data[i:i+block_size] for i in tensorRand])
    y = torch.stack([data[i+1:i+block_size+1] for i in tensorRand])
    x,y = x.to(device),y.to(device)
    return x,y
x, y = get_batch('train')
# print(x)
# print(y)

