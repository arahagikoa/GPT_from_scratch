import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' #  GPU
n_embd = 384
eval_iters = 200
n_head = 6
n_layer = 6
dropout = 0.2


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)



data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading

def get_batch(split):
    
    ''' 
    Generates a small batch of data of inputs x and targets y
    '''
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device) # move to CUDA device - GPU
    return x, y


@torch.no_grad() # we will not call it on backwards, we dont intend to do backprog here
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, 16)
        q = self.query(x) # (B, T, 16)


        # single head perform self-attention
        head_size = 16
        key = nn.Linear(C, head_size, bias = False)
        query = nn.Linear(C, head_size, bias = False)
        
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)


        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)

        return out

class feedForward(nn.Module):
    ''' simple linear layer followed by a non-linearity'''
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)

        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    ''' transformer block: communication followed by computation'''
    def __init__(self, n_ebmd, n_head) -> None:
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = feedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embeding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
       # self.sa_head = MultiHeadAttention(4, n_embd // 4) # 4 communication channels, 4 heads of 8 dimensional self-attention
        self.lm_head = nn.Linear(n_embd, vocab_size)
       # self.ffwd = feedForward(n_embd)
    def forward(self, idx, targets = None):
        B, T = idx.shape
    


        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B - batch, T - time, C - channel)
        pos_emb = self.position_embeding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) holds not only the tokens identity but also position 
        x = self.blocks(x)
        # x = self.sa_head(x)
        #x = self.ffwd(x)
        logits = self.lm_head(x) # (B, T, C)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # to match cross entropy 
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)


        return logits, loss # return scores for the next character in the squence, we are predicting what comes next
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):


            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            # applying softmax gets us probabilities
            probs = F.softmax(logits, dim = -1) # (B, C)

            # sample from sitribution
            idx_next = torch.multinomial(probs, num_samples= 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)

        return idx
    

model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
# === 300k tokens
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):


    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')


    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
