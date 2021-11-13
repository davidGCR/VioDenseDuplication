import cv2
import numpy as np

def prepare_image(image):
    """Prepare an image to extract edges with Canny

    Args:
        image (np.array/str): image or str of image

    Returns:
        image: image ready to process
    """
    
    # convert to RGB image and convert to float32
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def edgeDetection(image):
    """Performs Edge detection using Structured Forest

    Args:
        image (matrix): image to detect edges

    Returns:
        edges: image with edges
    """
    # initialize the structured edge detector with the model
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection('src/VioNet/Motioness/model.yml')
    # detect the edges
    edges = edge_detector.detectEdges(image)
    return edges

def edgeDetectionCanny(image):
    """Edge detectection using Canny algorithm

    Args:
        image (matrix): rgb image

    Returns:
        canny: image with edge detections
    """
    # grayscale and blurring for canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # carry out Canny edge detection
    canny = cv2.Canny(blurred, 50, 200)
    return canny

import add_libs
from dynamicimage.video_folder_2_di import process_frames_folder

if __name__=='__main__':
    l = 10
    segments = [
        list(range(i, i+10)) for i in range(0, 150, 10)
    ]
    

    video_folder = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames/train/Fight/_2RYnSFPD_U_0"

    for clip_indices in segments:
        clip_di = process_frames_folder(video_folder, None, indices=clip_indices)
        image = prepare_image(np.array(clip_di))
        edges = edgeDetection(image)

        cv2.imshow('di', np.array(clip_di))
        cv2.imshow('edges', edges)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break


    # img_path = '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet/dynamicimage/outputs/_q5Nwh4Z6ao_2.png'
    # image = cv2.imread(img_path)
    # # keep a copy of the original image
    # orig_image = image.copy()
    # image = prepare_image(image)
    
    # # canny = edgeDetectionCanny(orig_image)

    # # detect the edges
    # edges = edgeDetection(image)

    # save_name = 'src/VioNet/Motioness/outputs/_forests.jpg'
    # # cv2.imshow('DI', image)
    # # cv2.imshow('Canny', canny)
    # cv2.imshow('Structured forests', edges)
    # cv2.waitKey(0)