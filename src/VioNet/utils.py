import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.counter = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.counter += n
        self.avg = self.sum / self.counter


class Log(object):
    def __init__(self, path, keys):
        if os.path.exists(path):
            os.remove(path)
        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file, delimiter='\t')

        self.keys = keys
        self.writer.writerow(self.keys)

    def __del__(self):
        self.file.close()

    def log(self, values):
        v = []
        for key in self.keys:
            v.append(values[key])

        self.writer.writerow(v)
        self.file.flush()

def show_batch(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, epoch, optimizer, loss, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def load_checkpoint(model, device, optimizer, path):
    # checkpoint = torch.load(path)
    if device == torch.device('cpu'):
        checkpoint = torch.load(path, map_location=device)    
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss