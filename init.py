with open('tiny-shakespeare.txt') as f:
    text = f.read()

# do we have the right text?
print(text[:100])

# there are 65 possible tokens
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# encoders
# others: SentencePiece, tiktoken (subword)
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # string -> int[]
decode = lambda l: ''.join([itos[i] for i in l]) # int[] -> string

# pip3 install torch torchvision torchaudio
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# training, validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# aka context length
block_size = 8
print(train_data[:block_size+1])
# tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
# there are eight examples in this nine character sequence

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'when input is {context} the target: {target}')
# the transformer will receive at most eight characters to predict the next


torch.manual_seed(1337)
batch_size = 4 # num processed in parallel
block_size = 8

def get_batch(split):
    # returns: small batch of data, inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:', xb.shape)
print(xb)
print('targets:', yb.shape)
print(yb)

print('----')

# he's really spelling out how the processing works
# this is four parallel inputs
# each of size block_size
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'when input is {context.tolist()} the target: {target}')

# bigram language model
# covered in other videos

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)
        # reshape
        B, T, C = logits.shape
        # make one dimensional, make channel dimension the second dimension
        logits = logits.view(B*T, C)
        # make targets one dimensional. alternative: targets.view(-1)
        targets = targets.view(B*T)
        # how well are we predicting the next character? calculate loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus on previous time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
# torch.Size([32, 65])
# tensor(4.8786, grad_fn=<NllLossBackward0>)
# expecting loss to be -ln(1/65) or ~4.174
# not perfect

