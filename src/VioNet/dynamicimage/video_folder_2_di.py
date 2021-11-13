import os
from PIL import Image
import numpy as np
from transformations.dynamic_image_transformation import DynamicImage

def process_frames_folder(in_folder, out_file=None, indices=None):
    """Creates a dynamic image from a list of frames. Internally uses the DynamicImage transformation.
    
    Args:
        in_folder (str): path of the folder containing the frames
        out_file (str): path of the file to save the output in PNG format
        indices (list): list of indices indicating the frames to use to build the dynamic image
    
    Returns:
        output (PIL): dynamic image in PIL format

    """
    frames_names = os.listdir(in_folder)
    frames_names.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    frames_paths = [os.path.join(in_folder, frames_names[i]) for i in range(len(frames_names))]
    
    clip = [frames_paths[j] for j in indices]
 
    inputs = []
    for fp in clip:
        with Image.open(fp) as img:
            inputs.append(np.array(img.convert('RGB')))
    
    dynamic_image_fn = DynamicImage()
    output = dynamic_image_fn(inputs)
    if out_file:
        output.save(out_file, format="png")
    return output


    