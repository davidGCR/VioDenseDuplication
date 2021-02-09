import os
from PIL import Image
import numpy as np
import cv2
from  matplotlib import pyplot as plt
import glob

def dynamic_image_v1(frames, savePath = None):
    seqLen = len(frames)
    if seqLen < 2:
      print('No se puede crear DI con solo un frames ...', seqLen)
    frames = np.stack(frames, axis=0)
    fw = np.zeros(seqLen)  
    for i in range(seqLen): #frame by frame
      fw[i] = np.sum(np.divide((2 * np.arange(i + 1, seqLen + 1) - seqLen - 1), np.arange(i + 1, seqLen + 1)))
    # print('Di coeff=',fw)
    fwr = fw.reshape(seqLen, 1, 1, 1)  #coeficiebts

    # print('Coeficients v1:', fwr)

    sm = frames*fwr
    sm = sm.sum(0)
    sm = sm - np.min(sm)
    sm = 255 * sm / np.max(sm)
    img = sm.astype(np.uint8)
    ##to PIL image
    imgPIL = Image.fromarray(np.uint8(img))
    if savePath is not None:
      imgPIL.save(savePath)
    return imgPIL, img

def dynamic_image_v2(frames, normalized=True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    # print('channel_dynamic_images:', type(channel_dynamic_images))

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for channel in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames


def _compute_dynamic_image(frames):
    """ Adapted from https://github.com/hbilen/dynamic-image-nets """
    num_frames, h, w, depth = frames.shape

    # Compute the coefficients for the frames.
    coefficients = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

    # print('Coeficients v2:', coefficients)
    # Multiply by the frames by the coefficients and sum the result.
    x1 = np.expand_dims(frames, axis=0)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2

def get_video_frames(video_path):
    # Initialize the frame number and create empty frame list
    video = cv2.VideoCapture(video_path)
    frame_list = []

    # Loop until there are no frames left.
    try:
        while True:
            more_frames, frame = video.read()

            if not more_frames:
                break
            else:
                frame_list.append(frame)

    finally:
        video.release()

    return frame_list

if __name__ == '__main__':
    video_path = '/Users/davidchoqueluqueroman/Documents/CODIGOS/violencedetection2/DATASETS/RWF-2000/frames/train/Fight/_2RYnSFPD_U_0'
    frames_names = os.listdir(video_path)
    frames_names.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    frames = []
    for i in range(len(frames_names)):
        path = os.path.join(video_path, frames_names[i])
        # print(path)
        with Image.open(path) as img:
            # cv2.imshow('', np.array(img))
            # cv2.waitKey()

            # cv2.imshow('', np.array(img.convert('RGB')))
            # cv2.waitKey()

            # plt.imshow(img)
            # plt.show()

            # plt.imshow(img.convert('RGB'))
            # plt.show()

            frames.append(np.array(img.convert('RGB')))
    
    frames = frames[0:10]
    #v1
    imgPIL, img = dynamic_image_v1(frames)

    imgPIL.show()

    cv2.imshow('v1', img)
    cv2.waitKey()

    # print('frames:',type(frames), len(frames), type(frames[0]), frames[0].shape)

    ##v2
    # frames = glob.glob('{}/*.jpg'.format(video_path))
    # frames = [cv2.imread(f) for f in frames]

    # frames = frames[0:10]

    # dyn_image = dynamic_image_v2(frames, normalized=True)

    # print('dyn_image:', type(dyn_image))
    # cv2.imshow('', dyn_image)
    # cv2.waitKey()

    