import torch

# torch.manual_seed(123)
embedding = torch.nn.Embedding(6, 3)
print(embedding.weight)