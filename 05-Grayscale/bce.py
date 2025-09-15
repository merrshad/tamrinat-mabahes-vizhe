from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

recolored_img = Image.open('./images/AI_image.png').convert("RGB")
original_img = Image.open("./images/before_gray(Origin).jpg").convert("RGB")

target_size = (256, 256)
recolored_img = recolored_img.resize(target_size)
original_img = original_img.resize(target_size)


def to_tensor(img): return torch.tensor(
    np.array(img).transpose(2, 0, 1), dtype=torch.float32) / 255.0


recolored_tensor = to_tensor(recolored_img)
original_tensor = to_tensor(original_img)

recolored_tensor = recolored_tensor.unsqueeze(0)
original_tensor = original_tensor.unsqueeze(0)

bce_loss = F.binary_cross_entropy(recolored_tensor, original_tensor)

print("BCE Loss:", bce_loss.item())
