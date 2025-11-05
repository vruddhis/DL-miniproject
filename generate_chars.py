
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

df = pd.read_csv("characters.csv")

df["text"] = df.apply(
    lambda r: f"Name: {r['name']}\nElement: {r['element']}\nWeapon: {r['weapon']}\nRarity: {r['rarity']}\nDescription: {r['description']}\nPrompt: {r['prompt']}\n",
    axis=1
)
genshin_text = "\n".join(df["text"].tolist())


with open("fantasy_corpus.txt", "r", encoding="utf-8") as f:
    fantasy_text = f.read()

full_text = fantasy_text + "\n" + genshin_text

chars = sorted(list(set(full_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])

data_full = torch.tensor(encode(full_text), dtype=torch.long)

block_size = 100

def get_batch(data, batch_size=32):
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

model = LSTMModel(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def train_model(model, data, epochs=20, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        x, y = get_batch(data, batch_size)
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


train_model(model, torch.tensor(encode(fantasy_text), dtype=torch.long), epochs=50, lr=0.003)
torch.save(model.state_dict(), "pretrained.pt")


model.load_state_dict(torch.load("pretrained.pt"))
train_model(model, torch.tensor(encode(genshin_text), dtype=torch.long), epochs=30, lr=0.001)
torch.save(model.state_dict(), "genshin_finetuned.pt")



ELEMENTS = ["Anemo","Pyro","Hydro","Electro","Cryo","Geo","Dendro"]
WEAPONS = ["Sword","Bow","Polearm","Claymore","Catalyst"]
RARITIES = ["4-star","5-star"]

def sample_with_temperature(preds, temperature=0.8):
    preds = preds.float() / temperature
    probs = torch.softmax(preds, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def generate_character(model, start="Name:", length=600, temperature=0.8):
    model.eval()
    input_eval = torch.tensor([stoi[c] for c in start]).unsqueeze(0)
    hidden = None
    output = start

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_eval, hidden)
            next_idx = sample_with_temperature(logits[0,-1], temperature)
            next_char = itos[next_idx]
            output += next_char
            input_eval = torch.tensor([[next_idx]])
    return output


num_samples = 10
generated = []

for _ in range(num_samples):
    raw = generate_character(model, start="Name:", length=600, temperature=0.8)
    
    element = random.choice(ELEMENTS)
    weapon = random.choice(WEAPONS)
    rarity = random.choices(RARITIES, weights=[0.7,0.3])[0]
    raw = raw.replace("Element:", f"Element: {element}", 1)
    raw = raw.replace("Weapon:", f"Weapon: {weapon}", 1)
    raw = raw.replace("Rarity:", f"Rarity: {rarity}", 1)

    generated.append(raw)

pd.DataFrame({"generated_text": generated}).to_csv("genshin_generated.csv", index=False)


