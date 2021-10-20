
from functools import partial
import numpy as np

import cv2
import torch

# import detectron2
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection # Another option is slowfast_r50_detection

# from visualization import VideoVisualizer


device = 'cpu' # or 'cpu'
video_model = slow_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)

# Load the video
encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path('theatre.webm')
print('Completed loading encoded video.')

# Video predictions are generated at an internal of 1 sec from 90 seconds to 100 seconds in the video.
time_stamp_range = range(90,100) # time stamps in video for which clip is sampled. 
clip_duration = 1.0 # Duration of clip used for each inference step.
gif_imgs = []

for time_stamp in time_stamp_range:    
    print("Generating predictions for time stamp: {} sec".format(time_stamp))
    
    # Generate clip around the designated time stamps
    inp_imgs = encoded_vid.get_clip(
        time_stamp - clip_duration/2.0, # start second
        time_stamp + clip_duration/2.0  # end second
    )
    inp_imgs = inp_imgs['video']
    print('inp_imgs: ', type(inp_imgs), inp_imgs.size())
    
    # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
    # We use the the middle image in each clip to generate the bounding boxes.
    inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
    inp_img = inp_img.permute(1,2,0)
    
    # # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
    # predicted_boxes = get_person_bboxes(inp_img, predictor) 
    # if len(predicted_boxes) == 0: 
    #     print("Skipping clip no frames detected at time stamp: ", time_stamp)
    #     continue
        
    # # Preprocess clip and bounding boxes for video action recognition.
    # inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
    # # Prepend data sample id for each bounding box. 
    # # For more details refere to the RoIAlign in Detectron2
    # inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
    
    # # Generate actions predictions for the bounding boxes in the clip.
    # # The model here takes in the pre-processed video clip and the detected bounding boxes.
    # if isinstance(inputs, list):
    #     inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
    # else:
    #     inputs = inputs.unsqueeze(0).to(device)
    # preds = video_model(inputs, inp_boxes.to(device))

    # preds= preds.to('cpu')
    # # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
    # preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
    
    # # Plot predictions on the video and save for later visualization.
    # inp_imgs = inp_imgs.permute(1,2,3,0)
    # inp_imgs = inp_imgs/255.0
    # out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
    # gif_imgs += out_img_pred

print("Finished generating predictions.")