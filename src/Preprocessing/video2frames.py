import cv2
import numpy as np
from pathlib import Path
import os


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


def main():
    # data_path = Path("/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/hmdb51_org")
    # out_path = Path("/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/frames")
    
    data_path = Path("/media/david/datos/Violence DATA/RealLifeViolenceDataset/video")
    out_path = Path("/media/david/datos/Violence DATA/RealLifeViolenceDataset/frames")
    ext_ = ['*.avi', '*.mp4']

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # Locate folder with the HMDB51 data.
    data_path = Path(data_path)
    print(f'Loading HMDB51 data from [{data_path.resolve()}]...')

    # Iterate over each category (sub-folder).
    categories = list(data_path.glob('*/'))
    print(categories, len(categories))

    for subfolder in categories:
        # Make output sub-folder for each category.
        out_category_subfolder = out_path / subfolder.stem
        if not out_category_subfolder.exists():
            out_category_subfolder.mkdir()

        # Iterate over each video in the category and extract the frames.
        video_paths = subfolder.glob(ext_[1])
        for video_path in video_paths:
            # Create an output folder for that video's frames.   
            out_frame_folder = out_category_subfolder / video_path.st em

            if out_frame_folder.exists():
                continue
            out_frame_folder.mkdir()
            print('out_frame_folder: ', out_frame_folder)

            # Save the frames of the video. This process could be accelerated greatly by using ffmpeg if
            # available.
            # cmd = f'ffmpeg -i "{video_path}" -vf fps={fps} -q:v 2 -s {target_resolution[1]}x{target_resolution[0]} "{output_dir / "%06d.jpg"}"'
            
            frames = get_video_frames(str(video_path))
            for i, frame in enumerate(frames):
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(str(out_frame_folder / (str(i).zfill(6) + '.jpg')), frame)


if __name__ == '__main__':
    main()