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
    make_annotation_file(root="/Volumes/TOSHIBA EXT/DATASET/RWF-2000/train",
                         file_out="rwf-200-train_ann.txt",
                         dataset="rwf-2000")

