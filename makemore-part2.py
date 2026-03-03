#this is my implementation of Karpathy's makemore-part2 - character level language model - MLP implementation
#no AI used in this project.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

stoi = {s:i for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

block_size = 3
x, y = [], []

for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix1 = stoi[ch]
        x.append(context)
        y.append(ix1)
        context = context[1:] + [ix1]


x = torch.tensor(x)
y = torch.tensor(y)

c = torch.randn((27,2))

emb = c[x]
w1 = torch.randn((6,100)) * 0.1 
b1 = torch.randn((100)) * 0.1

h = torch.tanh(emb.view(emb.shape[0], 6)) @ w1 + b1

w2 = torch.randn((100, 27)) * 0.1 
b2 = torch.randn((27)) * 0.1

logits = h @ w2 + b2
counts = logits.exp()
probs = counts / counts.sum(1, keepdim = True)

loss = -probs[torch.arange(y.shape[0]), y].log().mean()
print(loss)