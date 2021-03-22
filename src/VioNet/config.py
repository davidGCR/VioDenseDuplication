class Config:

    def __init__(
            self,
            model,
            dataset,
            device,
            num_cv='',
            sample_duration=16,
            stride=1,
            sample_size=(112, 112),
            ft_begin_idx=3,
            acc_baseline=0.92,
            train_batch=32,
            val_batch=32,
            learning_rate=1e-3,
            momentum=0.5,
            weight_decay=1e-3,
            factor=0.1,
            min_lr=1e-7,
            num_epoch=1000,
            input_mode='rgb',
            train_temporal_transform = 'standar',
            val_temporal_transform = 'standar',
            segment_size = 2,
            additional_info = "",
            temp_annotation_path="",
            pretrained_model = None,
            num_classes=2,
    ):

        # VioNet models
        # resnext, densenet
        self.model = model

        # VioDB
        # hockey, movie, vif, mix
        self.dataset = dataset

        # Cross-validation
        # 1, 2, 3, 4, 5
        self.num_cv = num_cv

        # torch.device
        self.device = device

        # data
        self.sample_duration = sample_duration
        self.stride = stride
        self.sample_size = sample_size

        # finetune
        self.ft_begin_idx = ft_begin_idx

        # acc baseline
        self.acc_baseline = acc_baseline

        # batch size
        self.train_batch = train_batch
        self.val_batch = val_batch

        # optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # scheduler
        self.factor = factor
        self.min_lr = min_lr

        # training epoch
        self.num_epoch = num_epoch

        self.input_mode = input_mode 
        self.train_temporal_transform = train_temporal_transform
        self.val_temporal_transform = val_temporal_transform
        self.segment_size = segment_size
        self.additional_info = additional_info
        self.temp_annotation_path = temp_annotation_path
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
