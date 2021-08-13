import torch
from utils import AverageMeter
from itertools import cycle

def train_2it(_loader_violence, _loader_nonviolence, _epoch, _model, _criterion, _optimizer, _device, _num_tubes, _accuracy_fn):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(zip(_loader_violence, _loader_nonviolence)):
        video_images = torch.cat([data[0][1], data[1][1]], dim=0).to(_device)
        
        # id = 0
        # for j in range(data[0][0].size(0)):
        #     data[0][0][0,0] = id
        #     id += 1
        # for j in range(data[1][0].size(0)):
        #     data[1][0][0,0] = id
        #     id+=1
        boxes = torch.cat([data[0][0], data[1][0]], dim=0).to(_device)
        labels = torch.cat([data[0][2], data[1][2]], dim=0).to(_device)
        keyframes = torch.cat([data[0][5], data[1][5]], dim=0).to(_device)
        
        

        # print('video_images: ', video_images.size())
        # print('keyframes: ', keyframes.size())
        # print('num_tubes: ', _num_tubes)
        # print('boxes: ', boxes, boxes.size())
        # print('labels: ', labels, labels.size())

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(video_images, keyframes, boxes, _num_tubes)
        # print('outs: ', outs, outs.size())
        #loss
        loss = _criterion(outs,labels)
        #accuracy
        acc = _accuracy_fn(outs, labels)

        # print('batch: ', outs.size(), outs.shape[0])
        
        # meter
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()

        # len_data = min(len(_loader_violence), len(_loader_nonviolence))
        # print(
        #     'Epoch: [{0}][{1}/{2}]\t'
        #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #     'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #         _epoch,
        #         i + 1,
        #         len_data,
        #         loss=losses,
        #         acc=accuracies
        #     )
        # )
    train_loss = losses.avg
    train_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss.avg:.4f}\t'
        'Acc(train): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
    )
    return train_loss, train_acc

def val_2it(_loader_violence, _loader_nonviolence, _epoch, _model, _criterion, _device, _num_tubes, _accuracy_fn):
    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(zip(_loader_violence, _loader_nonviolence)):
        video_images = torch.cat([data[0][1], data[1][1]], dim=0).to(_device)
        boxes = torch.cat([data[0][0], data[1][0]], dim=0).to(_device)
        labels = torch.cat([data[0][2], data[1][2]], dim=0).to(_device)
        keyframes = torch.cat([data[0][5], data[1][5]], dim=0).to(_device)
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(video_images, keyframes, boxes, _num_tubes)
            loss = _criterion(outputs, labels)
            
            acc = _accuracy_fn(outputs, labels)
            
        losses.update(loss.item(), outputs.shape[0])
        accuracies.update(acc, outputs.shape[0])
    val_loss = losses.avg
    val_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss:.4f}\t'
        'Acc(val): {acc:.3f}'.format(_epoch, loss=val_loss, acc=val_acc)
    )

    return val_loss, val_acc

