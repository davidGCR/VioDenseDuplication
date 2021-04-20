import os

def make_annotation_file(root, file_out, dataset, video_formats=['avi', 'mp4']):
    if dataset == "UCFCrime2Local":
        # classes = os.listdir(root) #arrest, normal, vandalism, ...
        # classes = os.listdir(root)
        video_ann = []
        # for cl in classes:
        for cl in os.scandir(root):
            # print(cl)
            if cl.is_dir():
                videos = os.listdir(os.path.join(root,cl))
                video_names = [os.path.join(cl, v) for v in videos if v[0]!='.']
                for vn in video_names:
                    if not any([format in vn for format in video_formats]):
                            continue
                    print(vn)
                    video_ann.append(vn)

        with open(file_out, 'w') as fp:
            for name in video_ann:
                fp.write(name + '\n')
    if dataset == "UCFCrime2LocalClips":
        # classes = os.listdir(root) #arrest, normal, vandalism, ...
        # classes = os.listdir(root)
        video_ann = []
        
        for i, r in enumerate(root):
            videos = os.listdir(r)
            videos = [vid  for vid in videos if os.path.isdir(os.path.join(r,vid))]
            if i==1:
                videos = [vid for vid in videos if "Normal" in vid]
            for v in videos:
                if v[0]!='.':
                    label = "Normal" if "Normal" in v else "Abnormal"
                    video_ann.append(os.path.join(label, v+".mp4"))

        with open(file_out, 'w') as fp:
            for name in video_ann:
                fp.write(name + '\n')
    elif dataset == "rwf-2000":
        video_ann = []
        for cl in os.scandir(root):
            if cl.is_dir():
                videos = os.listdir(os.path.join(root,cl))
                video_names = [os.path.join(cl.name, v) for v in videos if v[0]!='.']
                for vn in video_names:
                    if not any([format in vn for format in video_formats]):
                            continue
                    print(vn)
                    video_ann.append(vn)

        with open(file_out, 'w') as fp:
            for name in video_ann:
                fp.write(name + '\n')


if __name__ == "__main__":
    make_annotation_file(root=("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips",
                                '/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames'),
                         file_out="UCFCrime2LocalClips-train_ann.txt",
                         dataset="UCFCrime2LocalClips")

