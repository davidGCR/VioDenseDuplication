import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')
import transformations.temporal_transforms as ts
import transformations.dynamic_image_transformation as tt
import random as rng

import cv2
import numpy as np
import os
import operator
from visual_utils import color, imread


class MotionSegmentation:
    def __init__(self, 
                        # video_detections,
                        # dataset_root,
                        config
                        # ratio_box_mmap,
                        # size=5,
                        # segment_size=5,
                        # stride=1,
                        # overlap=0,
                        # position='start'
                        ):
        # self.video_detections = video_detections
        # self.dataset_root = dataset_root
        # self.tmp_sampler = ts.SegmentsCrop(size=size,
        #                                     segment_size=segment_size,
        #                                     stride=stride,
        #                                     overlap=overlap,
        #                                     position=position)
        self.tmp_transform = tt.DynamicImage(output_type='ndarray')
        self.config = config
        # self.binary_thres = self.config['binary_thres']
        self.img_shape = (224,224)
        self.min_blob_area = 30
        self.score = 0
        # self.ratio_box_mmap = ratio_box_mmap
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
        # for det_box in frame_detections:
        #     ratio = self.ratio_motionmap_bbox(motion_map, det_box)
        #     # print('ratio:', ratio)
        #     if ratio>=self.ratio_box_mmap:
        #         # det_box[4] += self.score
        #         # print('score with motion: ', det_box[4])
        #         results.append(det_box)
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
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area > self.config['min_conected_comp_area']:
                (cX, cY) = centroids[i]
                final_components.append({
                    'id':i,
                    'x1':x,
                    'y1':y,
                    'x2':x + w,
                    'y2':y + h,
                    'cx': cX, 
                    'cY': cY,
                    'area': area
                })
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        
        return image, final_components

    def get_brightest_region(self, image, gray, radius):
        # apply a Gaussian blur to the image then find the brightest
        # region
        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        im = image.copy()
        cv2.circle(im, maxLoc, radius, (0, 255, 0), 2)
        # cv2.circle(im, minLoc, radius, (255, 0, 0), 2)
        return im
    
    def get_k_brightest_regions(self, image, gray, radius, k):
        """
        Get k brightest pixels from an image
        args:
            * image: rgb (h,w,3) to plot centers
            * gray: gray (h,w)
            * radius: kernel size for gaussian blur
            * k: number of pixels to return
        return:
            * im: rgb (h,w,3) with k ploted circles
            * centers: list of k coordinates of brightest pixels [(x,y), (x,y), ...]
        """
        # apply a Gaussian blur to the image then find the brightest
        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        idx = np.argpartition(-gray.ravel(),k)[:k]
        idx2=np.argsort(-gray.ravel()[idx])
        row,col = np.unravel_index(idx[idx2], gray.shape)
        xy = np.stack((row,col), axis=1)
        im = image.copy()
        centers = []
        for i,j in xy:
            centers.append((j,i))
            cv2.circle(im, (j,i), radius, (255, 0, 0), 1)
        return im, centers
    
    def get_k_dark_regions(self, image, gray, radius, k):
        """
        Get k darkness pixels from an image
        args:
            * image: rgb (h,w,3) to plot centers
            * gray: gray (h,w)
            * radius: kernel size for gaussian blur
            * k: number of pixels to return
        return:
            * im: rgb (h,w,3) with k ploted circles
            * centers: list of k coordinates of darkness pixels [(x,y), (x,y), ...]
        """
        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        idx = np.argpartition(gray.ravel(),k)[:k]
        idx2=np.argsort(gray.ravel()[idx])
        row,col = np.unravel_index(idx[idx2], gray.shape)
        xy = np.stack((row,col), axis=1)
        im = image.copy()
        centers = []
        for i,j in xy:
            centers.append((j,i))
            cv2.circle(im, (j,i), radius, (0, 0, 255), 1)
        return im, centers

    def color_quantization(self, image, K):
        Z = image.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)

        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        return res2, center

    def get_k_better_components(self, components, centers, k):
        max_component_idx = []
        counters = []
        for idx, c in enumerate(components):
            counter = 0
            for x,y in centers:
                insideComponent = x >= c['x1'] and y >= c['y1'] and x <= c['x2'] and y <= c['y2']
                counter = counter+1 if insideComponent else counter + 0
            counters.append(counter)
        
        s = np.array(counters)
        sort_index = np.argsort(s)
        sort_index.tolist()
        # print('sort_index: ', sort_index)
        sorted_components = [components[idx] for idx in sort_index]
        
        best_k_components = sorted_components[-k:]
        for i, b in enumerate(best_k_components):
            best_k_components[-i]['id'] = i

        # print('best_k_components: ', best_k_components)
        return best_k_components

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
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #(224,224)
            ##Threshold
            ret, thresh1 = cv2.threshold(gray, 0.5, 1, cv2.THRESH_BINARY)

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
    
    def non_motion_suppresion(
        self, 
        current_frame, 
        dyn_image_norm, 
        bright_pixels, 
        dark_pixels):
        im = np.array(current_frame)
        im_without_mov = np.abs(im*dyn_image_norm)
        im_without_mov = self.normalize(im_without_mov)
        gray = cv2.cvtColor(im_without_mov, cv2.COLOR_BGR2GRAY)
        ##Threshold
        ret, thresh1 = cv2.threshold(gray, self.config['binary_thres_norm'], 1, cv2.THRESH_BINARY)
        # apply connected component analysis to the thresholded image
        image_componets, components = self.conected_components(im, thresh1)
        best_components = self.get_k_better_components(components, bright_pixels + dark_pixels, self.config['k_best_components'])

        return im_without_mov, image_componets, best_components

    def motion_di_online(self, images, img_paths):
        plot = self.config['plot_config']['plot']
        wait = self.config['plot_config']['wait']
        save_results = self.config['plot_config']['save_results']
        save_folder = self.config['plot_config']['save_folder']

        # print(img_paths)
        #Motion image
        dyn_image = self.tmp_transform(img_paths)
        dyn_image_raw = dyn_image.copy()
        dyn_image_equ, centers = self.color_quantization(dyn_image, self.config['num_clusters_color_quantization'])
        ##Threshold
        gray_dyn_image = cv2.cvtColor(dyn_image, cv2.COLOR_BGR2GRAY) #(224,224)
        ret, thresh1 = cv2.threshold(gray_dyn_image, self.config['binary_thres'], 255, cv2.THRESH_BINARY)
        ##brightest regions
        brightest_image, bright_pixels = self.get_k_brightest_regions(
            dyn_image, 
            gray_dyn_image, 
            self.config['blur_kernel_size'], 
            self.config['k_brightnes_darkness_pixels'])
        brightest_image, dark_pixels = self.get_k_dark_regions(
            brightest_image, 
            gray_dyn_image, 
            self.config['blur_kernel_size'], 
            self.config['k_brightnes_darkness_pixels'])
        
        dyn_image = dyn_image_equ
        dyn_image_norm = self.normalize(dyn_image)

        if plot:
            # cv2.imshow('dyn_image', dyn_image_raw)
            # cv2.imshow('dyn_image_equ', dyn_image_equ)
            # cv2.imshow('dyn_image_norm', dyn_image_norm)
            # cv2.imshow('brightest_image', brightest_image)
            results_0 = [dyn_image_raw, dyn_image_equ, np.uint8(dyn_image_norm), brightest_image]
            # dtypes = [i.dtype for i in results_0]
            # print('dtypes: ', dtypes)
            cv2.imshow('results0', cv2.hconcat(results_0))

            if save_results:
                if not os.path.isdir(save_folder):
                    os.mkdir(save_folder)
                cv2.imwrite(save_folder + '/{}.jpg'.format('dyn_image'), dyn_image)
                cv2.imwrite(save_folder + '/{}.jpg'.format('dyn_image_equ'), dyn_image_equ)
                cv2.imwrite(save_folder + '/{}.jpg'.format('dyn_image_norm'), 255*dyn_image_norm)
                cv2.imwrite(save_folder + '/{}.jpg'.format('brightest_image'), brightest_image)
            
            key = cv2.waitKey(wait)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 
       
        if isinstance(images, list):
            motion_regions_map = []
            # motion_tube = []
            for idx, im in enumerate(images):
                frame_name = img_paths[idx].split('/')[-1]
                # print('---',frame_name)
                raw_image = im.copy()
                im_without_mov, image_componets, best_components = self.non_motion_suppresion(im, dyn_image_norm, bright_pixels, dark_pixels)
                mr = {
                    'frame': frame_name,
                    'top_k_m_regions': best_components
                }
                # print('{} m_regions in frame {}: \n {}'.format(len(best_components), frame_name, mr))
                motion_regions_map.append(mr)
                # if len(best_components)>0:
                #     # best_motion_region = best_components[0]
                #     # motion_regions_map['top_1_m_region'].append(best_motion_region)
                #     motion_regions_map['frames'].append(frame_name)
                #     print('{} m_regions in frame {}'.format(len(best_components), motion_regions_map['frames'][-1]))
                #     # for b_m_r in best_components:
                #     #     motion_regions_map['top_k_m_regions'].append()
                #     motion_regions_map['top_k_m_regions'].append(best_components)
                #     # print('best_motion_region: ', best_motion_region['x1'],best_motion_region['y1'],best_motion_region['x2'],best_motion_region['x2'])
                # # else:
                # #     motion_regions_map['frames'].append(img_paths[idx].split('/')[-1])
                # #     motion_regions_map['m_regions'].append(best_motion_region)

                if plot:
                    frame_name = img_paths[idx].split('/')[-1][:-4]
                    # cv2.imshow('frame_name', im)
                    for bc in best_components:
                        # print('bc: ', bc)
                        cv2.rectangle(
                            image_componets, 
                            (int(bc['x1']), int(bc['y1'])), 
                            (int(bc['x2']), int(bc['y2'])), 
                            color['deep pink'], 
                            3)
                        cv2.putText(image_componets, str(bc['id']), (int(bc['x1']),int(bc['y1']) - 7), cv2.FONT_ITALIC, 0.5, color['deep pink'], 3)
                    # cv2.imshow(frame_name + '_components', image_componets)
                    cv2.imshow('_without_mov', im_without_mov)
                    results_1 = [im, image_componets]
                    # dtypes = [i.dtype for i in results_1]
                    # print('dtypes2: ', dtypes)
                    cv2.imshow('results', cv2.hconcat(results_1))
                    if save_results:
                        cv2.imshow(frame_name + '_components', image_componets)
                        cv2.imshow(frame_name + '_without_mov', im_without_mov)
                        cv2.imwrite(save_folder + '/{}.jpg'.format(frame_name), raw_image)
                        cv2.imwrite(save_folder + '/{}.jpg'.format(frame_name + '_components'), image_componets)
                        cv2.imwrite(save_folder + '/{}.jpg'.format(frame_name + '_without_mov'), 255*im_without_mov)
                    # else:
                    #     cv2.imshow('_components', image_componets)
                    #     cv2.imshow('_without_mov', im_without_mov)

                    key = cv2.waitKey(wait)
                    if key == 27:#if ESC is pressed, exit loop
                        cv2.destroyAllWindows()

            # cv2.imshow('thresh1', thresh1)
            if plot:
                key = cv2.waitKey(wait)
                if key == 27:#if ESC is pressed, exit loop
                    cv2.destroyAllWindows() 
            # print('total  motion regions: ', len(motion_regions_map['top_k_m_regions']))
            return motion_regions_map



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
            ret, thresh1 = cv2.threshold(img, self.config['binary_thres'], 255, cv2.THRESH_BINARY)
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

    def __call__(self, images, img_paths):
        motion_map = self.motion_di_online(images, img_paths)
        # motion_map = self.motion_from_dynamic_images(frames)
        # motion_map = self.motion_from_background_substraction(frames)
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

from tube_config import *
from visual_utils import imread
if __name__=='__main__':
    segmentator = MotionSegmentation(MOTION_SEGMENTATION_CONFIG)
    video_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/violentflows/frames/1/Violence/fans_violence__F_GHT_S_MORE_MPORTANT_THAN_FOOTBALL_sometimes__shevchenko776__9a616Ppoeak'
    frames = os.listdir(video_path)
    
    num_frames = len(frames)
    print('num_frames: ', num_frames)
    numbers = np.linspace(0, num_frames, dtype=np.int16).tolist()
    print('numbers: ', numbers, len(numbers))

    def split_by_windows(frames, window_len):
        for i in range(0, len(frames), window_len): 
            yield frames[i:i + window_len]
    
    clips = list(split_by_windows(numbers, 10))
    print('clips: ', clips, len(clips))

    for clip in clips:
        print('------', clip)
        frames_paths = [os.path.join(video_path, 'frame{:03}.jpg'.format(i+1)) for i in clip]
        print('------', frames_paths)
        images = [np.array(imread(f)) for f in frames_paths]
        segmentator(images, frames_paths, True, save_folder=None)