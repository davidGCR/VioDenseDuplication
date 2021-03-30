import os

def make_annotation_file(root, file_out, dataset, video_formats=['avi', 'mp4']):
    if dataset == "UCFCrime2Local":
        classes = os.listdir(root) #arrest, normal, vandalism, ...
        video_ann = []
        for cl in classes:
            videos = os.listdir(os.path.join(root,cl))
            video_names = [os.path.join(cl, v) for v in videos if v[0]!='.']
            for vn in video_names:
                if not any([format in vn for format in video_formats]):
                        continue
                video_ann.append(vn)

        with open(file_out, 'w') as fp:
            for name in video_ann:
                fp.write(name + '\n')

if __name__ == "__main__":
    make_annotation_file(root="/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/videos",
                         file_out="test_ann.txt",
                         dataset="UCFCrime2Local")

