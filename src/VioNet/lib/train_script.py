import torch
from utils import AverageMeter

def train(_loader, _epoch, _model, _criterion, _optimizer, _device, _config, _accuracy_fn):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(_loader):
        # boxes, video_images, labels, num_tubes, paths, key_frames = data
        # boxes, video_images = boxes.to(_device), video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
       

        video_images, labels, paths, key_frames = data
        video_images = video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        boxes = None

        # print('video_images: ', video_images.size())
        # print('key_frames: ', key_frames.size())
        # print('boxes: ', boxes,  boxes.size())
        # print('labels: ', labels, labels.size())

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(video_images, key_frames, boxes, _config.num_tubes)
        #loss
        # print('labels: ', labels, labels.size(),  outs, outs.size())
        loss = _criterion(outs, labels)
        #accuracy
        acc = _accuracy_fn(outs, labels)
        
        # meter
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()
        print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                _epoch,
                i + 1,
                len(_loader),
                loss=losses,
                acc=accuracies
            )
        )
        
    train_loss = losses.avg
    train_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss.avg:.4f}\t'
        'Acc(train): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
    )
    return train_loss, train_acc

def val(_loader, _epoch, _model, _criterion, _device, _config, _accuracy_fn):
    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    for _, data in enumerate(_loader):
        # boxes, video_images, labels, num_tubes, paths, key_frames = data
        # boxes, video_images = boxes.to(_device), video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)

        video_images, labels, paths, key_frames = data
        video_images = video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        boxes = None
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(video_images, key_frames, boxes, _config.num_tubes)
            loss = _criterion(outputs, labels)
            acc = _accuracy_fn(outputs, labels)

        losses.update(loss.item(), outputs.shape[0])
        accuracies.update(acc, outputs.shape[0])

    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss.avg:.4f}\t'
        'Acc(val): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
    )
    val_loss = losses.avg
    val_acc = accuracies.avg

    return val_loss, val_acc

