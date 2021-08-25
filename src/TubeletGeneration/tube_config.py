MOTION_SEGMENTATION_CONFIG = {
    'binary_thres': 150,
    'min_conected_comp_area': 49,
    'num_clusters_color_quantization': 5,
    'plot_wait': 3000,
    'blur_kernel_size': 11,
    'k_brightnes_darkness_pixels': 100, #100
    'binary_thres_norm': 0.5,
    'k_best_components': 3,
}

TUBE_BUILD_CONFIG = {
    'dataset_root': '',
    'person_detections': '',
    'temporal_window': 5,
    'min_iou_close_persons': 0.3,
    'jumpgap': 5,
}