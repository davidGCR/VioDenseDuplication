from torch.utils.tensorboard import SummaryWriter

root = 'root'
writer = SummaryWriter(root)
for i in range(1, 100):
    writer.add_scalars('loss', {'train': 1 / i}, i)

# for i in range(1, 100):
    writer.add_scalars('loss', {'valid': 2 / i}, i)
