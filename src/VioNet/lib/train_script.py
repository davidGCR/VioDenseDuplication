import torch
from utils import AverageMeter, get_number_from_string

def train(_loader, _epoch, _model, _criterion, _optimizer, _device, _config, _accuracy_fn, _verbose=False):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
       

        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None

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
        if _verbose:
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
        'Loss(train): {loss:.4f}\t'
        'Acc(train): {acc:.3f}'.format(_epoch, loss=train_loss, acc=train_acc)
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
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)

        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(video_images, key_frames, boxes, _config.num_tubes)
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

def train_regressor(
    _loader, 
    _epoch, 
    _model, 
    _criterion, 
    _optimizer, 
    _device, 
    _config, 
    _accuracy_fn=None, 
    _verbose=False):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.float().to(_device)
        key_frames = key_frames.to(_device)
       

        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None

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
        if _accuracy_fn is not None:
            acc = _accuracy_fn(outs, labels)
        else:
            acc = 0
        
        # meter
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()
        if _verbose:
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

from customdatasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
from torchvision import transforms
from TubeletGeneration.tube_utils import JSON_2_videoDetections
from TubeletGeneration.tube_config import *
from TubeletGeneration.tube_test import extract_tubes_from_video
from TubeletGeneration.metrics import st_iou

def val_regressor(val_make_dataset):
    paths, labels, annotations, annotations_p_detections, num_frames = val_make_dataset()
    i = 0
    threshold = 0.2
    y_true = []
    pred_scores = []
    for path, label, annotation, annotation_p_detections, n_frames in zip(paths, labels, annotations, annotations_p_detections, num_frames):
        # if i > 0 :
        #     break
        print('{}--video:{}, num_frames: {}'.format(i+1, path, n_frames))
        print('----annotation:{}, p_detec: {}, {}'.format(annotation, annotation_p_detections, type(annotation_p_detections)))
        video_dataset = UCFCrime2LocalVideoDataset(
            path=path,
            sp_annotation=annotation,
            transform=transforms.ToTensor(),
            clip_len=n_frames,
            clip_temporal_stride=5
        )
        person_detections = JSON_2_videoDetections(annotation_p_detections)
        # print('person_detections: ', person_detections,len(person_detections))
        TUBE_BUILD_CONFIG['dataset_root'] = '/media/david/datos/Violence DATA/UCFCrime2LocalClips/UCFCrime2LocalClips'
        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        for clip, frames, gt, num_frames, frames_name in video_dataset:
            print('frAMES: ', frames.size())
            lps_split = extract_tubes_from_video(
                                    clip,
                                    MOTION_SEGMENTATION_CONFIG,
                                    TUBE_BUILD_CONFIG
                                    # gt=gt
                                    )
            real_numbers = [get_number_from_string(name) for name in lps_split[0]['frames_name']]
            print('real frame numbers:', real_numbers)
            # print('frames_name:', frames_name)                                                                        
            # print('extracted tubes: ', len(lps_split), lps_split[0]['frames_name'], lps_split[0]['foundAt'], ' max_num_frames: ', lps_split[0]['foundAt'][-1])
            video_dataset.get_center_frames(lps_split, get_number_from_string(frames_name[-1]))
            # le = loc_error_tube_gt(lps_split[0],gt, threshold=0.5)
            # print('localization_error: ', le)
            stmp_iou = st_iou(lps_split[0], gt)
            # print('stmp_iou: ', stmp_iou)
            y_true.append('positive')
            pred_scores.append(stmp_iou)
        i+=1


def train_2d_branch(
    _loader,
    _epoch, 
    _model,
    _criterion, 
    _optimizer, 
    _device, 
    _config,
    _accuracy_fn, 
    _verbose=False):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes = boxes.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(key_frames, boxes, _config.num_tubes)
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
        if _verbose:
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
        'Loss(train): {loss:.4f}\t'
        'Acc(train): {acc:.3f}'.format(_epoch, loss=train_loss, acc=train_acc)
    )
    return train_loss, train_acc

def val_2d_branch(_loader, _epoch, _model, _criterion, _device, _config, _accuracy_fn):
    print('validation at epoch: {}'.format(_epoch))
    # set model to evaluate mode
    _model.eval()
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()
    for _, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes = boxes.to(_device)
        labels = labels.to(_device)
        key_frames = key_frames.to(_device)
        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = _model(key_frames, boxes, _config.num_tubes)
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