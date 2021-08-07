import time
import torch
from utils import AverageMeter


def train_regressor(
    epoch, 
    data_loader, 
    model, 
    criterion, 
    optimizer, 
    device,
    epoch_log):
    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # set model to training mode
    model.train()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        # print('VioDB inputs: ',inputs.size())
        # print('VioDB target: ',targets.size())
        inputs, targets = inputs.to(device), targets.to(device)
        data_time.update(time.time() - end_time)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # meter
        losses.update(loss.item(), inputs.size(0))

        # backward + optimize
        loss.backward()
        optimizer.step()

        # meter
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses
            )
        )

        # batch_log.log(
        #     {
        #         'epoch': epoch,
        #         'batch': i + 1,
        #         'iter': (epoch - 1) * len(data_loader) + (i + 1),
        #         'loss': losses.val,
        #         'lr': optimizer.param_groups[0]['lr']
        #     }
        # )

    epoch_log.log(
        {
            'epoch': epoch,
            'loss': losses.avg,
            'lr': optimizer.param_groups[0]['lr']
        }
    )
    return losses.avg, optimizer.param_groups[0]['lr']



def train(
    epoch, 
    data_loader, 
    model, 
    criterion, 
    optimizer, 
    device, 
    batch_log,
    epoch_log
):
    print('training at epoch: {}'.format(epoch))

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # set model to training mode
    model.train()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        # print('VioDB inputs: ',inputs.size())
        # print('VioDB target: ',targets.size())
        inputs, targets = inputs.to(device), targets.to(device)
        data_time.update(time.time() - end_time)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy_2(outputs, targets)

        # print("accuracy1:{:4f}".format(calculate_accuracy(outputs, targets)))
        # print("accuracy2:{:4f}".format(calculate_accuracy_2(outputs, targets)))

        # meter
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        # backward + optimize
        loss.backward()
        optimizer.step()

        # meter
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies
            )
        )

        batch_log.log(
            {
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            }
        )

    epoch_log.log(
        {
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']
        }
    )
    return losses.avg, accuracies.avg, optimizer.param_groups[0]['lr']


def val(epoch, data_loader, model, criterion, device, val_log=None):
    print('validation at epoch: {}'.format(epoch))

    # set model to evaluate mode
    model.eval()

    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()

    for _, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy_2(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss.avg:.4f}\t'
        'Acc(val): {acc.avg:.3f}'.format(epoch, loss=losses, acc=accuracies)
    )
    if val_log:
        val_log.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg, accuracies.avg


def train_3dcnn_2dcnn(loader, _epoch, _model, _criterion, _optimizer, device):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(loader):
        images, labels, keyframes = data
        images =images.to(device)
        labels = labels.to(device)
        keyframes = keyframes.to(device)
        # print('data: ', len(data))
        # print('data[0]: ', len(data[0]))
        # print('data[1]: ', len(data[1])) 
        # print('video_images: ', video_images.size())
        # print('num_tubes: ', config.num_tubes)
        # print('boxes: ', boxes.size())
        # print('labels: ', labels, labels.size())

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(images, keyframes)
        # print('outs: ', outs, outs.size())
        #loss
        loss = _criterion(outs,labels)
        #accuracy
        acc = calculate_accuracy_2(outs,labels)
        # meter
        # print('len(video_images): ', len(video_images), ' video_images.size(0):',video_images.size(0), ' preds.shape[0]:', preds.shape[0])
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()
    train_loss = losses.avg
    train_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss.avg:.4f}\t'
        'Acc(train): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
    )
    return train_loss, train_acc


def val_3dcnn_2dcnn(loader, _epoch, _model, _criterion, device):
    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(loader):
        images, labels, keyframes = data
        images =images.to(device)
        labels = labels.to(device)
        keyframes = keyframes.to(device)
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(images, keyframes)
            loss = _criterion(outputs, labels)
            acc = calculate_accuracy_2(outputs,labels)
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

def test():
    pass


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def calculate_accuracy_2(y_pred, y_true):
    # Inspired from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
    _, predicted = torch.max(y_pred, 1)
    acc = (predicted == y_true).sum().item() / len(y_pred)
    return acc
