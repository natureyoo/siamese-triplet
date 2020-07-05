import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
import pkg_resources
import os
import torch.distributed as dist
from detectron2.utils import comm


register_coco_instances("deepfashion2_train", {}, "/second/DeepFashion2/coco_format/instance_train.json", "/second/DeepFashion2/train/image")
register_coco_instances("deepfashion2_val", {}, "/second/DeepFashion2/coco_format/instance_val.json", "/second/DeepFashion2/val/image/")

config_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
cfg_file = pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", config_path))

cfg = get_cfg()
cfg.merge_from_file(cfg_file)
cfg.DATASETS.TRAIN = ("deepfashion2_train",)
cfg.DATASETS.TEST = ("deepfashion2_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = '../output/model_0029999.pth'
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

# model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()