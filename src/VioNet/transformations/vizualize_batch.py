import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2 
import numpy as np


def draw_rect(im, cords, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    im = im.copy()
    
    cords = cords[:,:4]
    cords = cords.reshape(-1,4)
    if not color:
        color = [255,255,255]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
                
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

def plot_clip(clip, boxes, grid=(2,2), title=''):
    fig = plt.figure(figsize=(8., 8.))
    # fig.suptitle(title, fontsize=16)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=grid,  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for i, (ax, im) in enumerate(zip(grid, clip)):
        # Iterating over the grid returns the Axes.
        # ax.imshow(im)
        # print('boxes[i]: ', boxes[i], boxes[i].shape)
        # box = boxes[i].reshape(1,-1)[:,1:5]
        # print('box: ', box, box.shape)
        box = boxes[i].reshape(1,-1)
        # box = boxes[i]
        # print('box: ', box, box.shape)
        ax.imshow(draw_rect(im, box))

    plt.show()

def plot_keyframe(im, box):
    img = draw_rect(im, box)
    plt.imshow(img)
    plt.show()