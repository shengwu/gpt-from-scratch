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


