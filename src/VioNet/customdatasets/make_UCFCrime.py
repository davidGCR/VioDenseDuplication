import os
import re
import pickle
import cv2
import numpy as np
import random
import shutil

class MakeUCFCrime():
    def __init__(
        self, 
        root, 
        sp_abnormal_annotations_file, 
        sp_normal_annotations_file, 
        action_tubes_path,
        train,
        ground_truth_tubes=True):
        self.root = root
        self.sp_abnormal_annotations_file = sp_abnormal_annotations_file
        self.sp_normal_annotations_file = sp_normal_annotations_file
        # self.path_person_detections = path_person_detections
        self.classes = ['normal', 'abnormal'] 
        self.subclasses = ['Fighting', 'Assault', 'Robbery']
        # self.abnormal = abnormal
        self.train = train
        self.split = 'train' if self.train else 'test'
        self.ground_truth_tubes = ground_truth_tubes
        self.action_tubes_path = action_tubes_path
    
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
        filename_abnormal = self.sp_abnormal_annotations_file
        infile = open(filename_abnormal,'rb')
        abnormal_dict = pickle.load(infile)
        infile.close()

        filename_normal = self.sp_normal_annotations_file
        infile = open(filename_normal,'rb')
        normal_dict = pickle.load(infile)
        infile.close()

        new_dict = abnormal_dict.copy()
        new_dict.update(normal_dict)
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

    def gt2tube(self, annotations_per_frame):
        """
        Converts ground-truth to tube format
        """
        live_paths = [
            {
                'frames_name': [],
                'boxes': [],
                'len': 0,
                'id': 0,
                'foundAt': [],
                'score': 1
            }
        ]
        for idx, gt in enumerate(annotations_per_frame):
            live_paths[0]['frames_name'].append(gt['frame'])
            live_paths[0]['boxes'].append(np.array([gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax'], 1]))
            # live_paths[0]['foundAt'].append(gt['number'])
            live_paths[0]['foundAt'].append(idx)
            # live_paths[0]['frames_name'].append(gt['number'])
            live_paths[0]['len'] += 1
        return live_paths
    
    def gen_rand_bbox(self, size=(224,224), w_min=90, h_min=100):
        xmin = np.random.uniform(50, size[0]/2)
        ymin = np.random.uniform(50, size[1]/2)
        xmax = np.random.uniform(xmin + w_min, size[0])
        ymax = np.random.uniform(ymin + h_min, size[0])
        # xmax = xmin + w_min
        # ymax = ymin + h_min
        bbox = [xmin, ymin, xmax, ymax]
        return bbox
    
    def split_sequential_clips(self, frames, n_sample):
        for i in range(0, len(frames), n_sample): 
            yield frames[i:i + n_sample]
    
    def gen_temp_split(self, abnormal_path, skip=16, n_sample=16):
        frames = os.listdir(abnormal_path)
        raw_frames_numbers = [self.get_number_from_string(f_name) for f_name in frames]
        raw_frames_numbers = sorted(raw_frames_numbers)
        # if len(raw_frames_numbers)==0:
        #     print("No frames at: ", abnormal_path)
        # print('raw_frames_numbers len: ', len(raw_frames_numbers), abnormal_path)
        if int(len(raw_frames_numbers)/skip)<n_sample:
            frames_numbers = np.linspace(raw_frames_numbers[0], raw_frames_numbers[-1], n_sample).astype(int)
        else:
            #select frames every skip=16
            frames_numbers = [raw_frames_numbers[i] for i in range(0,len(raw_frames_numbers), skip)]
            #get sequential frames
            frames_numbers = list(self.split_sequential_clips(frames_numbers, n_sample))
            #remove short clips
            frames_numbers = [clip for clip in frames_numbers if len(clip)>=n_sample]
            frames_numbers = random.sample(frames_numbers, 1)[0]
            # frames_numbers = sorted(frames_numbers)
        # print('frames_numbers len: ', frames_numbers, len(frames_numbers))

        return frames_numbers
    
    def generate_normal_spt_annotations_file(self, out_file):
        """
        Generates sp annotation file for a normal video
        """
        _, normal_paths = self.__get_list__()
        _normal_annotations_dict = {}
        for ap in normal_paths:
            frames_numbers = self.gen_temp_split(ap)
            [x1, y1, x2, y2] = self.gen_rand_bbox(size=(320,240))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            _anno2save=[]
            for fn in frames_numbers:
                _anno2save.append([
                    os.path.join(ap, 'image_{:06}.jpg'.format(fn)),
                    x1,
                    y1,
                    x2,
                    y2
                ])
            _normal_annotations_dict[ap.split('/')[-1]] = _anno2save
            print('\nnormal key:\t', ap.split('/')[-1], ', value: ', _anno2save)
        file_to_write = open(out_file, "wb")
        pickle.dump(_normal_annotations_dict, file_to_write)
    
    def create_reduced_dataset(self, folder_src, folder_dst):
        """
        Creates reduced dataset using sp annotations
        """
        abnormal_paths, normal_paths = self.__get_list__()
        paths = abnormal_paths + normal_paths
        gt_dict = self.__load_sp_ground_truth__()
        for ap in paths:
            assert os.path.isdir(ap), 'Folder does not exist!!!'
            path_splits = ap.split('/')
            video_imgs_folder = ap.split('/')[-1]
            print('\nvideo_imgs_folder: ', ap)
            dst_folder = os.path.join(folder_dst, path_splits[-3], path_splits[-2], video_imgs_folder)
            if not os.path.isdir(dst_folder):
                print('creating dst_folder: ', dst_folder)
                os.mkdir(dst_folder)
            else:
                continue
            gt = gt_dict[video_imgs_folder]
            for frame_gt in gt:
                src_path = frame_gt[0]
                real_frame_name = '{:06}.jpg'.format(self.get_number_from_string(src_path.split('/')[-1]))
                h,t = os.path.split(src_path)
                src_path = os.path.join(h, real_frame_name)
                # print('src_path: ', src_path)
                if not os.path.isfile(src_path):
                    print('ERROR: {} does not exist!!!: '.format(src_path))
                    frame_path_splits = src_path.split('/')
                    print('frame_path_splits: ', frame_path_splits)
                    clase = 'abnormal' if 'Anomaly_Test' in frame_path_splits or 'Anomaly_Train' in frame_path_splits else 'normal'
                    split = 'train' if 'Anomaly_Train' in frame_path_splits else 'test'
                    src_path = os.path.join(folder_src, split, clase, frame_path_splits[-2], '{:06}.jpg'.format(self.get_number_from_string(frame_path_splits[-1])))
                print('\tcopying: {}'.format(src_path))
                img_dst = os.path.join(dst_folder, src_path.split('/')[-1])
                shutil.copyfile(src_path, img_dst)



#    No frames at:  /Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime/frames/train/normal/Normal_Videos533_x264
#    No frames at:  /Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime/frames/train/normal/Normal_Videos425_x264
#    No frames at:  /Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime/frames/train/normal/Normal_Videos566_x264
    def __call__(self):
        abnormal_paths, normal_paths = self.__get_list__()
        paths = abnormal_paths + normal_paths
        num_frames = []
        # sp_gts = []
        action_tubes = []
        gt_dict = self.__load_sp_ground_truth__()
        for ap in paths:
            assert os.path.isdir(ap), 'Folder does not exist!!!'
            n = len(os.listdir(ap))
            num_frames.append(n)
            annotations_per_frame = []

            if self.ground_truth_tubes:
                gt = gt_dict[ap.split('/')[-1]]
                # print('\nkey:\t', ap.split('/')[-1], ', value: ', gt)
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
                # sp_gts.append(annotations_per_frame)
                action_tubes.append(self.gt2tube(annotations_per_frame))
            else:
                video_path_splits = ap.split('/')
                tube_annotation_path = os.path.join(self.action_tubes_path, video_path_splits[-3], video_path_splits[-2], video_path_splits[-1]+'.json')
                # print('tube_annotation_path: ', tube_annotation_path)
                if not os.path.isfile(tube_annotation_path):
                    print('\nAnnotation does not exist!!! ', tube_annotation_path)
                
                action_tubes.append(tube_annotation_path)



            # if ap.split('/')[-2] == 'abnormal':
            #     gt = gt_dict[ap.split('/')[-1]]
            #     # print('\nkey:\t', ap.split('/')[-1], ', value: ', gt)
            #     for g in gt:
            #         f_number=self.get_number_from_string(g[0].split('/')[-1])
            #         (x1, y1) = self.new_coordinates_after_resize_img((320,240), (224,224), (int(g[1]),int(g[2])))
            #         (x2, y2) = self.new_coordinates_after_resize_img((320,240), (224,224), (int(g[3]),int(g[4])))
            #         annotations_per_frame.append({
            #             "number": f_number,
            #             "frame": '{:06}.jpg'.format(f_number),
            #             "xmin": x1,
            #             "ymin": y1,
            #             "xmax": x2,
            #             "ymax": y2
                        
            #         })
            #     annotations_per_frame = sorted(annotations_per_frame, key = lambda i: i['number'])
            #     sp_gts.append(annotations_per_frame)
            #     gt_tubes.append(self.gt2tube(annotations_per_frame))
            # else:
            #     frames_numbers = self.gen_temp_split(ap)
            #     [x1, y1, x2, y2] = self.gen_rand_bbox((224,224))
            #     for fn in frames_numbers:
            #         annotations_per_frame.append({
            #             "number": fn,
            #             "frame": '{:06}.jpg'.format(fn),
            #             "xmin": x1,
            #             "ymin": y1,
            #             "xmax": x2,
            #             "ymax": y2
                        
            #         })
            #     annotations_per_frame = sorted(annotations_per_frame, key = lambda i: i['number'])
            #     sp_gts.append(annotations_per_frame)
            #     gt_tubes.append(self.gt2tube(annotations_per_frame))
            
        labels = [1]*len(abnormal_paths) + [0]*len(normal_paths)
        return paths, labels, action_tubes, num_frames

if __name__=='__main__':
    make_func = MakeUCFCrime(
        root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime_Reduced/frames', 
        sp_abnormal_annotations_file='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/UCFCrime/Train_annotation.pkl', 
        sp_normal_annotations_file='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/UCFCrime/Train_normal_annotation.pkl',
        action_tubes_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/UCFCrime_Reduced', 
        train=True,
        ground_truth_tubes=True)
    
    paths, labels, action_tubes, num_frames = make_func()
    rand_idx = random.randint(0,len(paths)-1)
    print('paths: ', len(paths), paths[rand_idx])
    print('labels: ', len(labels), labels[rand_idx])
    print('action_tubes: ', len(action_tubes))
    # print('gt_tubes: ', len(gt_tubes))
    print('num_frames: ', len(num_frames))

    # make_func.create_reduced_dataset(
    #     folder_src='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime/frames',
    #     folder_dst='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime_Reduced/frames'
    #     )

    # make_func.generate_normal_spt_annotations_file(
    #     "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/UCFCrime/Train_normal_annotation.pkl"
    #     )

    idx = 0
    print('action_tubes[idx]: ', action_tubes[idx], len(action_tubes[idx]))
    # print('sp_gts[idx]: ', sp_gts[idx], len(sp_gts[idx]))
    # print('\ngt_tubes[idx]: ', gt_tubes[idx], len(gt_tubes[idx][0]['frames_name']))
    # make_func.plot(folder_imgs=paths[idx], annotations_dict=sp_gts[idx], live_paths=[])