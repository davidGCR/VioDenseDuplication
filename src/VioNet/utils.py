
import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import re

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



class TimeMeter(object):
    def __init__(self):
        self.time = 0
        self.fps = 0
        self.total_time = 0
        self.frames_counter = 0

    def update(self, time, num_frames=1):
        self.time = time
        self.total_time += time
        self.frames_counter += num_frames
        self.fps = self.frames_counter/self.total_time
    
    def save_summary(self, dst_file):
        dc = {
            'total_time': self.total_time,
            'total_frames':  self.frames_counter,
            'fps': self.fps
        }
        with open(dst_file, 'w') as fp:
            json.dump(dc, fp)


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

def show_batch(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    if title:
        plt.title(title)
    plt.show()

def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, epochs, last_epoch, optimizer, loss, path):
    torch.save({
            'epochs': epochs,
            'last_epoch': last_epoch,
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
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    last_epoch = checkpoint['last_epoch']

    loss = checkpoint['loss']

    return model, optimizer, epochs, last_epoch, loss

def atoi(text):
            return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
def natural_sort(l: list):
    l.sort(key=natural_keys)
    return l

class colors:
# '''Colors class:reset all colors with colors.reset; two
# sub classes fg for foreground
# and bg for background; use as colors.subclass.colorname.
# i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
# underline, reverse, strike through,
# and invisible work with the main class i.e. colors.bold'''
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'
    class fg:
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
    class bg:
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'
 
# print(colors.bg.green, "SKk", colors.fg.red, "Amartya")
# print(colors.bg.lightgrey, "SKk", colors.fg.red, "Amartya")

def get_number_from_string(f_name):
    return int(re.search(r'\d+', f_name).group())

if __name__== '__main__':
    device = get_torch_device()
    print(colors.reset, "DEVICE: ", device)