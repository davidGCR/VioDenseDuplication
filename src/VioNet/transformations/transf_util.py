from torchvision import transforms
import numpy as np
from PIL import Image

def tensor2PIL(t):
    im = transforms.ToPILImage()(t).convert("RGB")
    return im

def PIL2tensor(img):
    im = transforms.ToTensor()(img)
    return im

def PIL2numpy(img):
    return np.array(img)

def imread(path):
    with Image.open(path) as img:
        return img.convert('RGB')