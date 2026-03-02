#This is my replication of Andrej Karpathy's makemore project to understand how language models work, and are built from scratch.
#There is no AI or auto complete used in this project.

import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
N = torch.zeros(27,27, dtype = torch.int32)
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

p = N[0].float()
p = p / p.sum()
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=20, replacement=True, generator = g)


#copied visualization code from jupyter notebook
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

plt.show()