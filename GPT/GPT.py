import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import math
import time


if torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'
print(device)

iterations = 4000
training = False
#------------------

@dataclass
class Config:
  block_size: int = 64 # max sequence length
  vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
  n_layer: int = 4 # number of layers
  n_head: int = 4 # number of heads
  n_embd: int = 128 # embedding dimension
  dropout = 0.1


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = iterations * 0.25
max_steps = iterations

lr_list = []
loss_list = []
val_loss_list = []

total_batch_size = 4096
B = 8 # micro batch size
T = Config.block_size # sequence length
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)

class DataLoader():
  def __init__(self, B, T, split):
    self.B = B
    self.T = T

    with open("GPT/input.txt", "r", encoding="utf-8") as file:
      text = file.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    if split == "train":
      self.tokens = self.tokens[:int(0.9 * len(self.tokens))]
    else:
      self.tokens = self.tokens[int(0.9 * len(self.tokens)):]
    print(f"len of {split} tokens: {len(self.tokens)}")
    self.current_position = 0
  def next_batch(self):
    B, T = self.B, self.T
    buffer = self.tokens[self.current_position : self.current_position +B *T +1 ]

    x = (buffer[:-1]).view(B,T)
    y = (buffer[1:]).view(B,T)
    self.current_position += B*T
    if self.current_position + (B*T +1) > len(self.tokens):
      self.current_position = 0
    x, y = x.to(device), y.to(device)
    return x,y
    

class CasualSelfAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.dropout = nn.Dropout(config.dropout)
    self.c_proj.GPT_SCALE_INIT = 1
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
    q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
    v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.c_proj(y)
    y = self.dropout(y)
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
    self.dropout = nn.Dropout(config.dropout)
    self.c_proj.GPT_SCALE_INIT = 1

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.sa = CasualSelfAttention(config)
    self.feed_forward = MLP(config)
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.feed_forward(self.ln2(x))
    return x



class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd),
      position_embedding_table = nn.Embedding(config.block_size, config.n_embd),
      blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
      ln_f = nn.LayerNorm(config.n_embd),
      ))
    
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

    self.transformer.token_embedding_table.weight = self.lm_head.weight

    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, "GPT_SCALE_INIT"):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    assert T <= self.config.block_size, f"Cannot forward sequence because block size is only {self.config.block_size}"

    pos = torch.arange(0, T, dtype=torch.long, device = device)
    
    tok_emb = self.transformer.token_embedding_table(idx)
    
    pos_emb = self.transformer.position_embedding_table(pos).to(device)
    x = tok_emb + pos_emb
    for block in self.transformer.blocks:
      x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))

    return logits, loss

  def generate(self, text, num_sequences, max_length):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)  # encode text
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)  # repeat for num_sequences
    x = tokens.to(device)

    output = ["" for i in range(num_sequences)]

    generated_length = tokens.size(1)
    
    while generated_length < max_length:
        # Ensure that the input does not exceed block_size for each iteration
        # Take the last block_size tokens (or fewer if fewer tokens exist)
        input_tokens = x[:, -self.config.block_size:] 

        with torch.no_grad():
            logits, loss = self(input_tokens)  # forward pass
            
            # Get the logits of the last token
            logits = logits[:, -1, :]
            
            # Get the probabilities using softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # sample from top-k
            ix = torch.multinomial(topk_probs, 1)  # sample next token
            xcol = torch.gather(topk_indices, -1, ix)  # get the actual token index
            
            # Append the generated token to the sequence
            x = torch.cat((x, xcol), dim=1)
            generated_length += 1  # increase the generated length

    # Decoding the generated tokens
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        output[i] = decoded
        # print(output[i])
    return output
  

  def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
     


def get_lr(iteration):
   
    if iteration < warmup_steps:
        return max_lr * (iteration+1) / warmup_steps
    
    if iteration > max_steps:
        return min_lr
    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(model, optimizer, iteration, loss_list, val_list, lr_list, checkpoint_path):
  checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'loss_list': loss_list,
        'val_list': val_list,
        'lr_list': lr_list
    }
  torch.save(checkpoint, checkpoint_path)
  print(f"Checkpoint saved at iteration {iteration} to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model state

    if optimizer:  # Load optimizer state if provided (for resuming training)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iteration = checkpoint['iteration']
    loss_list = checkpoint.get('loss_list', [])  # Load loss list (default to empty if not present)
    val_loss_list = checkpoint.get('val_list', [])  # Load validation loss list if available
    lr_list = checkpoint.get('lr_list', [])  # Load learning rate history

    print(f"Checkpoint loaded from iteration {iteration}")
    return iteration, loss_list, val_loss_list, lr_list




train_loader = DataLoader(B, T, split="train")
val_loader = DataLoader(B, T, split="val")


model = GPT(Config)
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = model.configure_optimizers(0.1, 6e-4, device)



done_iteration, loss_list, val_loss_list, lr_list = load_checkpoint("GPT/checkpoint_epoch_3999.pth", model, optimizer)
# done_iteration = 1
if training:
  for iter in range(done_iteration, iterations):

    
    t0 = time.time()

    if iter % 100 == 0 or iter == iterations -1:
      model.eval()
      with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.item()
        val_loss_list.append(val_loss_accum)
      checkpoint_path = f'GPT/GPTlogs/checkpoint_epoch_{iter}.pth'
      save_checkpoint(model, optimizer, iter, loss_list, val_loss_list, lr_list, checkpoint_path)
      print("Val loss: " + str(val_loss_accum))
      model.generate("The Queen: ", 5, 50)

    model.train()

    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      logits, loss = model(x, y)
      loss = loss / grad_accum_steps
      loss_accum += loss.item()
      loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(iter)
    lr_list.append(lr)
    loss_list.append(loss_accum)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    difT = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
  
    print(f"step {iter}, loss: {loss_accum}, norm: {norm:.2f}, lr: {lr} time: {difT:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")

if not training:
  
  model.eval()
  with torch.no_grad():
    val_loss_accum = 0.0
    val_loss_steps = 20
    for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    print("Val loss: " + str(val_loss_accum))



import matplotlib.pyplot as plt

# Create a figure and subplots
fig, axs = plt.subplots(2, figsize=(10, 6))  # Adjust figure size for better spacing
fig.suptitle('Training and Validation Loss', fontsize=16, weight='bold', y=1.05)  # Title for the entire figure

# Plotting Loss list
axs[0].plot(loss_list, color='b', linestyle='-', linewidth=2)
axs[0].set_title("Training Loss", fontsize=14)
axs[0].set_xlabel('Epochs', fontsize=12)
axs[0].set_ylabel('Loss', fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.6)  # Add gridlines

# Plotting Validation Loss list
axs[1].plot(val_loss_list, color='r', linestyle='-', linewidth=2)
axs[1].set_title("Validation Loss", fontsize=14)
axs[1].set_xlabel('Epochs', fontsize=12)
axs[1].set_ylabel('Loss', fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.6)  # Add gridlines

# Tighten the layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to give space for suptitle

# Save the figure
plt.savefig('GPT/loss_plot.png', dpi=300, bbox_inches='tight')  # Save as PNG with high DPI for quality

# Show the plot
plt.show()



# f = open("output.txt", "w")
# f.write(model.generate("The Queen: ", 1, 10000)[0])
# f.close()