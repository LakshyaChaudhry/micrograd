#this is my implementation of Karpathy's makemore-part2 - character level language model - MLP implementation
#no AI used in this project.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

block_size = 3

def build_dataset(words):
    x, y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix1 = stoi[ch]
            x.append(context)
            y.append(ix1)
            context = context[1:] + [ix1]
    return torch.tensor(x), torch.tensor(y)

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

xtr, ytr = build_dataset(words[:n1])
xdev, ydev = build_dataset(words[n1:n2])
xte, yte = build_dataset(words[n2:])

c = torch.randn((27,10))

emb = c[xtr]
w1 = torch.randn((30,300))
b1 = torch.randn((300))

h = torch.tanh(emb.view(emb.shape[0], 30) @ w1 + b1)

w2 = torch.randn((300, 27)) 
b2 = torch.randn((27)) 

parameters = [c, w1, b1, w2, b2]

for p in parameters:
    p.requires_grad_(True)

# lre = torch.linspace(-3, 0, 1000)
# lrs =  10*lre
# lri = []
# lrs = []


for i in range(50000):
    ix = torch.randint(0, xtr.shape[0], (32,))
    emb = c[xtr[ix]]
    h = torch.tanh(emb.view(emb.shape[0], 30) @ w1 + b1)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

# evaluate on all splits
for name, xs, ys in [('train', xtr, ytr), ('dev', xdev, ydev), ('test', xte, yte)]:
    emb = c[xs]
    h = torch.tanh(emb.view(emb.shape[0], 30) @ w1 + b1)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, ys)
    print(f'{name} loss: {loss.item():.4f}')

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = c[torch.tensor([context])]
        h = torch.tanh(emb.view(1, 30) @ w1 + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out[:-1]))