from .ops import paste_masks_in_image
from .masks import BitMasks
import detectron2

detectron2.layers.paste_masks_in_image = paste_masks_in_image
detectron2.structures.BitMasks = BitMasks

import os
import json
import torch
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.exposure import rescale_intensity

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNN
from detectron2.config import CfgNode as CN

from .models import MultiMaskRCNNConvUpsampleHead as MultiR
from .postprocessing import postproc_multimask
from .utils import initialize_new_config_values

class YeastMatePredictor():
    def __init__(self, cfg, weights=None):
        self.cfg = get_cfg()
        
        self.cfg = initialize_new_config_values(self.cfg)

        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = 'cpu'

        self.cfg.merge_from_file(cfg)

        if weights is not None:
            self.cfg.MODEL.WEIGHTS = weights

        self.model = GeneralizedRCNN(self.cfg)
        self.model.to(torch.device(self.cfg.MODEL.DEVICE))
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

    @staticmethod
    def image_to_tensor(image):
        height, width = image.shape

        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)

        image = torch.as_tensor(image)  
        image = {"image": image, "height": height, "width": width}

        return image

    def preprocess_img(self, image, norm=True, zstack=False):
        if zstack:
            image = image[image.shape[0]//2]

        if len(image.shape) > 2:
            image = image[:,:,0]

        if norm:
            image = image.astype(np.float32)
            lq, uq = np.percentile(image, [1.5, 98.5])
            image = rescale_intensity(image, in_range=(lq,uq), out_range=(0,1))
        else:
            image = image.astype(np.float32)

        image = self.image_to_tensor(image)

        return image

    def detect(self, image):
        with torch.no_grad():
            return self.model([image])[0]['instances']

    def inference(self, image, zstack=False, norm=True):

        image = self.preprocess_img(image, zstack=zstack, norm=norm)

        instances = self.detect(image)

        possible_comps = self.cfg.POSTPROCESSING.POSSIBLE_COMPS
        optional_object_score_threshold = self.cfg.POSTPROCESSING.OPTIONAL_OBJECT_SCORE_THRESHOLD
        parent_override_threshold = self.cfg.POSTPROCESSING.PARENT_OVERRIDE_THRESHOLD

        things, mask = things, mask = postproc_multimask(instances, possible_comps, \
            optional_object_score_threshold=optional_object_score_threshold, parent_override_thresh=parent_override_threshold)

        return things, mask

    def inference_on_folder(self, folder, zstack=False, norm=True):

        #### EXTEND THIS FOR FULL FUNCTIONALITY

        pathlist = glob(os.path.join(folder, '/*.tif')) + glob(os.path.join(folder, '/*.tiff'))

        for path in folder:
            things, mask = inference(imread(path), zstack=zstack, norm=norm)

            resdict = {'image': os.path.basename(path), 'metadata': {}, 'detections': things}

            imsave(path.replace('.tif', '_mask.tif'), mask)
            with open(path.replace('.tif', '_detections.json'), 'w') as file:
                doc = json.dump(resdict, file, indent=1)

    @staticmethod
    def postprocess_instances(instances, possible_comps, optional_object_score_threshold=0.15, parent_override_threshold=2, score_thresholds={0:0.9, 1:0.5, 2:0.5}):
        possible_comps_dict = {}
        for n in range(len(possible_comps)):
            new_comps = {}
            for key in possible_comps[n]:
                new_comps[int(key)] = possible_comps[n][key]

            possible_comps_dict[n+1] = new_comps

        things, mask = postproc_multimask(instances, possible_comps_dict, \
            optional_object_score_threshold=optional_object_score_threshold, parent_override_thresh=parent_override_threshold, score_thresholds=score_thresholds)

        return things, mask
