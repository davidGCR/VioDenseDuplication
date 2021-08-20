# import sys
# sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src')
import transformations.temporal_transforms as ts
import transformations.dynamic_image_transformation as tt
import random as rng

import cv2
import numpy as np
import os
from PIL import Image
def imread(path):
    with Image.open(path) as img:
        return img.convert('RGB')

class MotionSegmentation:
    def __init__(self, video_detections,
                        dataset_root,
                        ratio_box_mmap,
                        size=5,
                        segment_size=5,
                        stride=1,
                        overlap=0,
                        position='start'
                        ):
        self.video_detections = video_detections
        self.dataset_root = dataset_root
        self.tmp_sampler = ts.SegmentsCrop(size=size,
                                            segment_size=segment_size,
                                            stride=stride,
                                            overlap=overlap,
                                            position=position)
        self.tmp_transform = tt.DynamicImage(output_type='ndarray')
        self.binary_thres = 150
        self.img_shape = (224,224)
        self.min_blob_area = 30
        self.score = 0
        self.ratio_box_mmap = ratio_box_mmap
        # self.processed_img = None
    
    def blob_detection(self, gray_image):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 200;
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(gray_image)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(gray_image,
                                                keypoints,
                                                np.array([]),
                                                (0,0,255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        key = cv2.waitKey(5000)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows() 

    def total_motion_area(self, areas):
        total_area = 0
        for a in areas:
            total_area += a
        return total_area

    def ratio_motionmap_bbox(self, motion_map, box):
        """
        Compute the ratio = C/len(centers_mass), 
        where C is the number of blob centers into the 'box'.
        """
        ratio = 0
        total_int_area = 0
        box_area = (box[2]-box[0])*(box[3]-box[1])
        polygons = motion_map['polygons']
        for i in range(len(polygons)):
            int_area = self.intersect_area_poly_box(img_shape=self.img_shape, 
                                                        pol1=polygons[i], 
                                                        box=box
                                                        )
            total_int_area += int_area
        ratio = total_int_area/box_area
        return ratio

    def filter_no_motion_boxes(self, frame_detections, motion_map):
        results = []
        for det_box in frame_detections:
            ratio = self.ratio_motionmap_bbox(motion_map, det_box)
            # print('ratio:', ratio)
            if ratio>=self.ratio_box_mmap:
                # det_box[4] += self.score
                # print('score with motion: ', det_box[4])
                results.append(det_box)
        return results

    def polygon_from_blob(self, cnt):
        # approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        # approx = cv2.approxPolyDP(cnt, 0.01* cv2.arcLength(cnt, True), True)
        approx = cv2.approxPolyDP(cnt, 3, True)
        return approx
    
    def box_from_polygon(self, pol):
        approx = cv2.boundingRect(pol)
        return approx
    
    def DrawPolygon(self, ImShape,Polygon,Color):
        Im = np.zeros(ImShape, np.uint8)
        # try:
        #             cv2.fillPoly(Im, Polygon, Color) # Only using this function may cause errors, I don’t know why
        # except:
        #     try:
        #         cv2.fillConvexPoly(Im, Polygon, Color)
        #     except:
        #         print('cant fill\n')

        cv2.fillConvexPoly(Im, Polygon, Color)
    
        return Im

    def intersect_area_poly_box(self, img_shape, pol1, box):
        pol2 = np.array([[box[0],box[1]],
                        [box[0],box[1]+ box[3]-box[1]],
                        [box[2],box[3]],
                        [box[0] + box[2]-box[0],box[1]],
                        ]).astype(int)
        Im1 = self.DrawPolygon(img_shape,pol1,122)
        Im2 = self.DrawPolygon(img_shape,pol2,133)
        Im = Im1 + Im2
        ret, OverlapIm = cv2.threshold(Im, 200, 255, cv2.THRESH_BINARY)
        IntersectArea=np.sum(np.greater(OverlapIm, 0))

        # box_Area = (box[2]-box[0])*(box[3]-box[1])
        # print('BoxArea: {}, IntersectArea:{}'.format(box_Area, IntersectArea))
        
        # cv2.imshow('OverlapIm',OverlapIm)
        # key = cv2.waitKey(1000)
        # if key == 27:#if ESC is pressed, exit loop
        #     cv2.destroyAllWindows() 

        # print('IntersectArea:', IntersectArea)
        
        # contours, hierarchy = cv2.findContours(OverlapIm,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours)>0:
        #     contourArea=cv2.contourArea(contours[0])
        #     print('contourArea={}\n'.format(contourArea))
        #     perimeter = cv2.arcLength(contours[0], True)
        #     print('contourPerimeter={}\n'.format(perimeter))
        #     RealContourArea=contourArea+perimeter
        #     print('RealContourArea={}\n'.format(RealContourArea))
        return IntersectArea

    def contour_2_box(self, cnt):
        box = [int(cnt[0]), int(cnt[1]), int(cnt[0]+cnt[2]), int(cnt[1]+cnt[3])]
        return box
    
    def find_contour_areas(self, contours):
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
        return areas

    def remove_big_short_blobs(self, im, max_area_percent=0.7):
        mask = np.zeros(im.shape, np.uint8)
        frame_area = self.img_shape[0]*self.img_shape[1]
        contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i=0
        for j, area in enumerate(self.find_contour_areas(contours)):
            if area<=max_area_percent*frame_area and area>self.min_blob_area:
                # final_contours.append(area)
                # cv2.drawContours(mask, [cnt], i, (255,255,255), 1)
                cv2.fillPoly(mask, pts =[contours[j]], color=(255,255,255))
                i+=1
        return mask
    
    def split_video(self, frames):
        num_splits = 6
        split_size = 25

    def normalize(self, data):
        data = (data / 255.0).astype(np.float32)
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std
    
    def compute_mean_image(self, images, normalize=True):
        """
        input:
            frames: list of np.array images [(224,224,3), (224,224,3), ...]
        output:
            avg_sum: average image of frames
        """
        # images = [np.array(i) for i in images]
        images = np.array(images, dtype = np.float32) #(224, 224, 3)
        avg_sum = np.mean(images, axis=0)
        if normalize:
            avg_sum = self.normalize(avg_sum)
        return avg_sum

        
    def remove_background(self, image, avg_sum, normalize=False):
        image_without_backgorund = np.abs(image - avg_sum)
        if normalize:
            image_without_backgorund = self.normalize(image_without_backgorund)
        return image_without_backgorund
    
    def compute_motion_in_boxes(self, frame_without_back, detections):
        return None

    def conected_components(self, image, thresh1):
        output = cv2.connectedComponentsWithStats((thresh1*255).astype(np.uint8), 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        # loop over the number of unique connected component labels
        final_components = []
        for i in range(1, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            # if i == 0:
            #     text = "examining component {}/{} (background)".format(
            #         i + 1, numLabels)
            # # otherwise, we are examining an actual connected component
            # else:
            #     text = "examining component {}/{}".format( i + 1, numLabels)
            # print a status message update for the current connected
            # component
            # print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # ensure the width, height, and area are all neither too small
            # nor too big
            keepWidth = w > 5 and w < 50
            keepHeight = h > 45 and h < 65
            keepArea = area > 500 and area < 1500

            if area > 49:
                # final_components.append()
                (cX, cY) = centroids[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        
        return image

    def motion_from_background_substraction(self, frames):
        """
        input:
            frames: list of image indices [0, 6, 12, ...]
        
        """
        #read images
        img_paths, images = self.read_segment(frames)
        avg_sum = self.compute_mean_image(images, normalize=False)
        
        for i, im in enumerate(images):
            im = self.remove_background(im, avg_sum, True)
            ##Threshold
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #(224,224)
            ret, thresh1 = cv2.threshold(gray, 0.5, 1, cv2.THRESH_BINARY)
            # thresh1 = cv2.threshold((gray*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # kernel = np.ones((5,5),np.uint8)
            # thresh1 = cv2.dilate(thresh1,kernel,iterations = 1)

            # apply connected component analysis to the thresholded image
            image_boxes = self.conected_components(im,thresh1)

            cv2.imshow('{}'.format(i+1), im)
            cv2.imshow('thresh1-{}'.format(i+1), thresh1)
            cv2.imshow('componentes-{}'.format(i+1), image_boxes)
            key = cv2.waitKey(1000)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()

        #plot
        avg_sum = self.normalize(avg_sum)
        cv2.imshow('avg_sum', avg_sum)
        key = cv2.waitKey(3000)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
        
        return None
    
    def read_segment(self, segment):
        """
        input:
            list of frames paths
        output: (img_paths, images)
            img_paths: list of image paths
            images: list of np.array images [(224,224,3), (224,224,3), ...]
        """
        img_paths = []
        images = []
        for f in segment:
            split = self.video_detections[f]['split']
            video = self.video_detections[f]['video']
            frame = self.video_detections[f]['fname']
            img_path = os.path.join(self.dataset_root, split, video, frame)
            img_paths.append(img_path)
            images.append(np.array(imread(img_path)))
        return img_paths, images
    
    def motion_di_online(self, frames):
        segments = self.tmp_sampler(frames)
        motion_images = []
        clips_idxs = []
        wait = 3000
        for i, segment in enumerate(segments): #For each video clip process a set of dynamic images
            print('-----segment: ', i+1)
            img_paths, images = self.read_segment(segment)
            #Motion image
            dyn_image = self.tmp_transform(img_paths)
            cv2.imshow('dyn_image', dyn_image)
            dyn_image_norm = self.normalize(dyn_image)
            cv2.imshow('dyn_image_norm', dyn_image_norm)
            key = cv2.waitKey(wait)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 
            
            for im in images:
                im = np.array(im)
                cv2.imshow('im', im)
                im_without_mov = np.abs(im*dyn_image_norm)
                im_without_mov = self.normalize(im_without_mov)
                cv2.imshow('im_without_mov', im_without_mov)
                key = cv2.waitKey(wait)
                if key == 27:#if ESC is pressed, exit loop
                    cv2.destroyAllWindows()


            ##Threshold
            img = cv2.cvtColor(dyn_image, cv2.COLOR_BGR2GRAY) #(224,224)
            ret, thresh1 = cv2.threshold(img, self.binary_thres, 255, cv2.THRESH_BINARY)
            # thresh1 = self.remove_big_short_blobs(thresh1)
            # motion_images.append(thresh1)
            # clips_idxs += segment
            cv2.imshow('thresh1', thresh1)
            key = cv2.waitKey(wait)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 
        
        return None



    def motion_from_dynamic_images(self, frames):
        segments = self.tmp_sampler(frames)
        motion_images = []
        clips_idxs = []
        for i, segment in enumerate(segments): #For each video clip process a set of dynamic images
            print('-----segment: ', i+1)
            img_paths = []
            for f in segment:
                split = self.video_detections[f]['split']
                video = self.video_detections[f]['video']
                frame = self.video_detections[f]['fname']
                img_path = os.path.join(self.dataset_root, split, video, frame)
                print(img_path)
                img_paths.append(img_path)
            #Motion image
            dyn_image = self.tmp_transform(img_paths)
            ##Threshold
            img = cv2.cvtColor(dyn_image, cv2.COLOR_BGR2GRAY) #(224,224)
            ret, thresh1 = cv2.threshold(img, self.binary_thres, 255, cv2.THRESH_BINARY)
            thresh1 = self.remove_big_short_blobs(thresh1)
            motion_images.append(thresh1)
            clips_idxs += segment

        # avg_image = np.mean(motion_images, axis=0, dtype=np.float32)
        avg_image = sum(motion_images)#/len(motion_images)
        # self.blob_detection(avg_image)
        motion_map={
                        'img': avg_image,
                        'sub_motion_images': motion_images,
                        'frames': clips_idxs
                    }
        #compute contours, moments, and certer of mass
        processed_img = motion_map['img'] #(224,224)
        frames = motion_map['frames']
        #preprocess
        # kernel = np.ones((5,5),np.uint8)
        # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((5,5),np.uint8)
        processed_img = cv2.dilate(processed_img,kernel,iterations = 1)
        ################################
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print('areas:', self.find_contour_areas(contours))
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        # print('areas sorted:', self.find_contour_areas(contours))
        #area of blobs
        contours_areas = self.find_contour_areas(contours)
        #polygons from blobs
        polygons = []
        boxes_from_polygons = []
        for cnt in contours:
            approx = self.polygon_from_blob(cnt)
            polygons.append(approx)
            boxes_from_polygons.append(self.box_from_polygon(approx))

        
        # Get the moments
        mu = [None]*len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        # Get the mass centers
        mc = [None]*len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
        # save centers
        motion_map['mc'] = mc
        motion_map['processed_img'] = processed_img
        motion_map['contours'] = contours
        motion_map['areas'] = contours_areas
        motion_map['polygons'] = polygons
        motion_map['boxes_from_polygons']=boxes_from_polygons
        return motion_map

    def __call__(self,frames):
        # motion_map = self.motion_di_online(frames)
        # motion_map = self.motion_from_dynamic_images(frames)
        motion_map = self.motion_from_background_substraction(frames)
        return motion_map

    def plot(self, motion_map, lbbox=[], wait=300):
        processed_img = motion_map['processed_img']
        contours = motion_map['contours']
        mc = motion_map['mc']
        polygons = motion_map['polygons']
        boxes_from_polygons = motion_map['boxes_from_polygons']
        thresh1_rgb = np.dstack([processed_img]*3)
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(thresh1_rgb, contours, i, (0,0,255), 1)
            cv2.circle(thresh1_rgb, (int(mc[i][0]), int(mc[i][1])), 4, (0,255,0), -1)
            #draw polygon
            cv2.drawContours(thresh1_rgb, [polygons[i]], 0, (147,20,255), 2)
            #draw boxes_from_polygons
            cv2.rectangle(thresh1_rgb,
                            (int(boxes_from_polygons[i][0]), int(boxes_from_polygons[i][1])),
                            (int(boxes_from_polygons[i][0]+boxes_from_polygons[i][2]), int(boxes_from_polygons[i][1]+boxes_from_polygons[i][3])),
                            (0,238,238),
                            2)
        if len(lbbox) > 0:
            for bbox in lbbox:
                p1 = (int(bbox[0]),int(bbox[1]))
                p2 = (int(bbox[2]),int(bbox[3]))
                cv2.rectangle(thresh1_rgb, p1, p2, color, 3)
        cv2.imshow('avg_image', thresh1_rgb)
        key = cv2.waitKey(wait)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows() 
    
    def plot_sub_motion_imgs(self, motion_image, wait=500):
        sm_imgs = motion_image['sub_motion_images']
        for i, img in enumerate(sm_imgs):
            cv2.imshow('motion_image_{}'.format(i), img)
            key = cv2.waitKey(wait)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()