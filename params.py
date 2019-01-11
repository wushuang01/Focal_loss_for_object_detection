GET_TARGET_PARAMS=\
{
    #as the pre in the paper:
    "FPN_ANCHOR_SCALES":[24,48,96,192,384],
    "FPN_FEAT_STRIDES":[8,16,32,64,128],
    "FPN_ANCHOR_STRIDE":1,
    "ANCHOR_RATIOS":[0.5,1,2],
    "ANCHOR_SCALES ":[1.,pow(2,1/3.), pow(2,2/3.)],
    "POSITIVE_OVERLAP":0.5,
    "NEGTIVE_OVERLAP":0.4,
    #for voc dataset
    "classes":20,
    "confidence":0.4,
    "NMS_THRESH":0.3,
}
TRAINING_PARAMS=\
{
    "classes":20,
    "lr":{
        "backbone_lr":0.0001,
        "other_lr":0.001,
        "decay_gamma":0.1,
        "decay_step":30,
    },
    "epochs":160,
    "optimizer":
    {
        "type":"sgd",
        "weight_decay":4e-05,
    },
    "batch_size":16,
    "train_path":"/home/omnisky/PycharmProjects/yolo_snip/DOTA/new_PNGimage",
    # "train_path":"/home/omnisky/PycharmProjects/RetinaNet/data",
    "val_path":"/home/omnisky/PycharmProjects/RetinaNet/data",
    "img_h":384,
    "img_w":384,
    "parallels": [0],
    "model_save_dir": "/home/wushuanchen/PycharmProjects/RetinaNet/model/new_try2",
    "pretrain_snapshot":"",
}
TEST_PARAMS=\
{
    "classes":20,
    "batch_size":1,
    "train_path":"/home/omnisky/PycharmProjects/yolo_snip/DOTA/val/new_PNGimages",
    "img_h":384,
    "img_w":384,
    "parallels": [1],
    "images_path": "/home/wushuanchen/PycharmProjects/RetinaNet/test_images/",
    "pretrain_snapshot":"/home/wushuanchen/PycharmProjects/RetinaNet/model/new_try2/20190103215048model.pth",
    "classes_names_path":"/home/wushuanchen/PycharmProjects/RetinaNet/dataset/coco.names",
    'val_path':"/home/omnisky/PycharmProjects/yolo_snip/DOTA/val/new_PNGimages",
}