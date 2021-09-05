import numpy as np
import torch
from customdatasets.dataset_utils import imread
# from VioNet.customdatasets.dataset_utils import imread
from PIL import Image

class DynamicImage():
    def __init__(self, output_type="pil", savePath=None):
        self.savePath = savePath
        self.output_type = output_type
    
    def __to_tensor__(self, img):
        img = img.astype(np.float32)/255.0
        t = torch.from_numpy(img)
        t = t.permute(2,0,1)
        return t
    
    def __read_imgs__(self, pths):
        # for p in pths:
        #     _, v = os.path.split(p)
        #     print(v)
        frames = [imread(p) for p in pths]
        # frames = [np.array(imread(p)) for p in pths]
        return frames

    def __call__(self, frames):
        if isinstance(frames[0], str):
            frames = self.__read_imgs__(frames)
        elif torch.is_tensor(frames):
            frames = frames.numpy()
            frames = [f for f in frames]
        seqLen = len(frames)
        if seqLen < 2:
            print('No se puede crear DI con solo un frames ...', seqLen)
        
        # print(len(frames))
        # for f in frames:
        #     print('--', f.shape)
        try:
            frames = np.stack(frames, axis=0)
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print(len(frames))
            for f in frames:
                print('--', f.size)

        fw = np.zeros(seqLen)  
        for i in range(seqLen): #frame by frame
            fw[i] = np.sum(np.divide((2 * np.arange(i + 1, seqLen + 1) - seqLen - 1), np.arange(i + 1, seqLen + 1)))
        # print('Di coeff=',fw)
        fwr = fw.reshape(seqLen, 1, 1, 1)  #coeficiebts
        sm = frames*fwr
        sm = sm.sum(0)
        sm = sm - np.min(sm)
        sm = 255 * sm / np.max(sm)
        img = sm.astype(np.uint8)
        ##to PIL image
        imgPIL = Image.fromarray(np.uint8(img))
        if self.savePath is not None:
            imgPIL.save(self.savePath)
        if self.output_type == "ndarray":
            return img
        elif self.output_type == "pil":
            return imgPIL
        elif self.output_type == "tensor":
            return self.__to_tensor__(img)