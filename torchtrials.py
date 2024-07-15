import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.multinomial
probabilities = torch.tensor([0.1, 0.9])
sample = torch.multinomial(probabilities, num_samples=10, replacement=True)
# print(sample)
# torch.cat
tensor = torch.tensor([1,2,3,4])
new_tensor = torch.cat((tensor, torch.tensor([5])), dim=0)
# print(new_tensor)
# upper and lower triangles
hi1 = torch.tril(torch.ones(4,4))
# print(hi1)
# masked.fill function
hi2 = torch.zeros(5,5).masked_fill((torch.tril(torch.ones(5,5)) == 0), float('-inf'))
# print(hi2)
# print(torch.exp(hi2))
hi3 = torch.zeros(2,3,4)
hi4 = hi3.transpose(0,2)
# print(hi4.shape)
inpoo = torch.tensor([[10.,10.,10.],[5.,5.,5.],[3.,3.,3.]])
model1 = nn.Linear(3,3, bias=False)
# for param in model.parameters():
#     print(param)
# print(model(inpoo))
softie = F.softmax(inpoo, dim=1)
# print(softie)
# nn.Embedding
num_embeddings = 10
embedding_dim = 3
embedding = nn.Embedding(num_embeddings, embedding_dim)
tensor1 = torch.tensor([[2,4,6,9], [3,4,8,1]])
# print(tensor1)
tensor1 = embedding(tensor1)
# print(tensor1)
print(torch.randint(5, (2,3)))
