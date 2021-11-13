import add_libs
from video_folder_2_di import process_frames_folder


if __name__ == '__main__':
    in_folder = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames/train/Fight/_2RYnSFPD_U_0'
    out_ = 'src/VioNet/dynamicimage/outputs/'+in_folder.split('/')[-1]+'.png'

    dyn_img = process_frames_folder(
                                    in_folder=in_folder,
                                    out_file=out_,
                                    indices=list(range(0, 140)))