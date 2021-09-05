MOTION_SEGMENTATION_CONFIG = {
    'binary_thres': 150,
    'min_conected_comp_area': 49,
    'num_clusters_color_quantization': 5,
    'blur_kernel_size': 11,
    'k_brightnes_darkness_pixels': 100, #100
    'binary_thres_norm': 0.5,
    'k_best_components': 3,
    'plot_config':{
        'plot': False,
        'wait': 1000,
        'save_results': False, 
        'save_folder': None,
    },
}

TUBE_BUILD_CONFIG = {
    'dataset_root': '',
    'person_detections': '',
    'close_persons_rep': 10,#
    'temporal_window': 5,
    'min_iou_close_persons': 0.3,
    'jumpgap': 5,
    'plot_config':{
        'debug_mode': False,
        'plot': False,
        'plot_wait_1': 1000,
        'plot_wait_2':1000,
        'save_results': False,
        'save_folder': None
    }
    
}