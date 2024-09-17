import torch
import torch.nn as nn
from torch.nn import functional as F
import BigramModel as BigramModel

if torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'



with open("/Users/enderpeyzner/Downloads/CompSci/DeepLearning/GPT/input.txt", "r", encoding="utf-8") as file:
  text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)


model = BigramModel.BigramLanguageModel()
m = model.to(device)

model.load_state_dict(torch.load('BigramModel/model.pth', weights_only=True))
model.eval()


context = torch.zeros((1, 1), dtype=torch.long, device = device)

f = open("demofile2.txt", "w")
f.write(decode(model.generate(context, max_new=10000)[0].tolist()))
f.close()
