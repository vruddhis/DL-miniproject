import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

csv_path = "genshin_dataset/characters.csv"
output_dir = "genshin_dataset/lora_output"
base_model = "runwayml/stable-diffusion-v1-5"
lr = 1e-4
num_epochs = 1      
batch_size = 1
image_size = 512
max_train_steps = 200  
device = "mps" if torch.backends.mps.is_available() else "cpu"

os.makedirs(output_dir, exist_ok=True)


class GenshinDataset(Dataset):
    def __init__(self, csv_path, image_size=512):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB").resize((self.image_size, self.image_size))
        prompt = row["prompt"]
        return prompt, image

dataset = GenshinDataset(csv_path)

pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, config)

optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=lr)

for step, (prompt, image) in enumerate(tqdm(dataset, total=min(len(dataset), max_train_steps))):
    if step >= max_train_steps:
        break
    with torch.no_grad():
        latent = pipe.vae.encode(pipe.feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)).latent_dist.sample()
        latent = latent * 0.18215

    text_input = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length)
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    noise = torch.randn_like(latent)
    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=device).long()
    noisy_latent = pipe.scheduler.add_noise(latent, noise, timesteps)

    noise_pred = pipe.unet(noisy_latent, timesteps, encoder_hidden_states=text_embeddings).sample
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

torch.save(pipe.unet.state_dict(), os.path.join(output_dir, "lora_genshin.pt"))

