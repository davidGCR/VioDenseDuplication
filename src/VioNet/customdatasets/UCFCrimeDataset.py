import os
import re
import pickle
import cv2
import numpy as np

class MakeUCFCrime():
    def __init__(self, root, sp_annotations_file, train):
        self.root = root
        self.sp_annotations_file = sp_annotations_file
        # self.path_person_detections = path_person_detections
        self.classes = ['normal', 'abnormal'] 
        self.subclasses = ['Fighting', 'Assault', 'Robbery']
        # self.abnormal = abnormal
        self.train = train
        self.split = 'train' if self.train else 'test'
        # self.split_file = split_file
    
    def __get_list__(self):
        path = self.root
        
        abnormal_videos = os.listdir(os.path.join(path, self.split, self.classes[1]))
        abnormal_paths = [os.path.join(path, self.split, self.classes[1], video) for video in abnormal_videos if os.path.isdir(os.path.join(path, self.split, self.classes[1], video))]
        
        normal_videos = os.listdir(os.path.join(path, self.split, self.classes[0]))
        normal_paths = [os.path.join(path, self.split, self.classes[0], video) for video in normal_videos if os.path.isdir(os.path.join(path, self.split, self.classes[0], video))]

        return abnormal_paths, normal_paths
    
    def __annotation__(self, folder_path):
        v_name = os.path.split(folder_path)[1]
        annotation = [ann_file for ann_file in os.listdir(self.path_annotations) if ann_file.split('.')[0] in v_name.split('(')]
        annotation = annotation[0]
        # print('annotation: ',annotation)
        return os.path.join(self.path_annotations, annotation)
    
    def __load_sp_ground_truth__(self):
        filename = self.sp_annotations_file
        infile = open(filename,'rb')
        new_dict = pickle.load(infile)
        infile.close()
        return new_dict

    def new_coordinates_after_resize_img(self, original_size, new_size, original_coordinate):
        original_size = np.array(original_size)
        new_size = np.array(new_size)
        original_coordinate = np.array(original_coordinate)
        xy = original_coordinate/(original_size/new_size)
        x, y = int(xy[0]), int(xy[1])
        return (x, y) 
    
    def plot(self, folder_imgs, annotations_dict, live_paths=[]):
        imgs = os.listdir(folder_imgs)
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        
        imgs.sort(key=natural_keys)
        # print(type(folder_imgs),type(f_paths[0]))
        f_paths = [os.path.join(folder_imgs, ff) for ff in imgs]
        
        for img_path in f_paths:
            # print('img_path: ', img_path)

            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # f_num = img_path.split('/')[-1]
            # f_num = self.get_number_from_string(f_num)
            # print('f_num: ', f_num)
            frame = img_path.split('/')[-1]
            ann = [ann for ann in annotations_dict if ann['frame']==frame]
            # print('annotations_dict: ', annotations_dict)
            # print('ann: ', ann)
            if len(ann)>0:
                x1 = ann[0]["xmin"]
                y1 = ann[0]["ymin"]
                x2 = ann[0]["xmax"]
                y2 = ann[0]["ymax"]

                
                cv2.rectangle(image,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0,238,238),
                                1)
                # cv2.circle(image, (y1,y2), 2, (0,0,255), 1)
                if len(live_paths)>0:
                    frame = img_path.split('/')[-1]
                    
                    for l in range(len(live_paths)):
                        
                        foundAt = True if frame in live_paths[l]['frames_name'] else False
                        if foundAt:
                            idx = live_paths[l]['frames_name'].index(frame)
                            bbox = live_paths[l]['boxes'][idx]
                            x1 = bbox[0]
                            y1 = bbox[1]
                            x2 = bbox[2]
                            y2 = bbox[3]
                            cv2.rectangle(image,
                                        (int(x1), int(y1)),
                                        (int(x2), int(y2)),
                                        (255,0,0),
                                        1)
                size = (600,600)
                cv2.namedWindow('FRAME'+str(frame),cv2.WINDOW_NORMAL)
                cv2.resizeWindow('FRAME'+str(frame), size)
                image = cv2.resize(image, size)
                cv2.imshow('FRAME'+str(frame), image)
                key = cv2.waitKey(250)#pauses for 3 seconds before fetching next image
                if key == 27:#if ESC is pressed, exit loop
                    cv2.destroyAllWindows()

    def get_number_from_string(self, f_name):
        return int(re.search(r'\d+', f_name).group())

    def __call__(self):
        abnormal_paths, normal_paths = self.__get_list__()
        paths = abnormal_paths + normal_paths
        num_frames = []
        sp_gts = []
        gt_dict = self.__load_sp_ground_truth__()
        for ap in paths:
            assert os.path.isdir(ap), 'Folder does not exist!!!'
            n = len(os.listdir(ap))
            num_frames.append(n)
            annotations_per_frame = []
            if ap.split('/')[-2] == 'abnormal':
                gt = gt_dict[ap.split('/')[-1]]
                for g in gt:
                    f_number=self.get_number_from_string(g[0].split('/')[-1])
                    (x1, y1) = self.new_coordinates_after_resize_img((320,240), (224,224), (int(g[1]),int(g[2])))
                    (x2, y2) = self.new_coordinates_after_resize_img((320,240), (224,224), (int(g[3]),int(g[4])))
                    annotations_per_frame.append({
                        "number": f_number,
                        "frame": '{:06}.jpg'.format(f_number),
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2
                        
                    })
                annotations_per_frame = sorted(annotations_per_frame, key = lambda i: i['number'])

                sp_gts.append(annotations_per_frame)
            else:
                gt = None
                sp_gts.append(gt)
            # print('\ngt: ', sp_gts[-1])
        labels = [1]*len(abnormal_paths) + [0]*len(normal_paths)
        return paths, labels, sp_gts, num_frames

if __name__=='__main__':
    make_func = MakeUCFCrime(
        root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime/frames', 
        sp_annotations_file='Train_annotation.pkl', 
        train=True)
    
    paths, labels, sp_gts, num_frames = make_func()
    print('paths: ', len(paths), paths[0])
    print('labels: ', len(labels))
    print('sp_gts: ', len(sp_gts))
    print('num_frames: ', len(num_frames))

    idx = 21
    print('sp_gts[idx]: ', sp_gts[idx])
    make_func.plot(folder_imgs=paths[idx], annotations_dict=sp_gts[idx], live_paths=[])