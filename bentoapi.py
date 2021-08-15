import base64
import torch
import logging
import numpy as np
from io import BytesIO
from PIL import Image

import bentoml
from bentoml.adapters import ImageInput, AnnotatedImageInput
from bentoml.artifact import PytorchModelArtifact

from yeastmatedetector.inference import YeastMatePredictor

@bentoml.env(docker_base_image='yeastmate:trainer')
@bentoml.artifacts([PytorchModelArtifact('model')])
class YeastMate(bentoml.BentoService):
    @bentoml.api(input=AnnotatedImageInput(pilmode='F'))
    def predict(self, image, annotations):
        logging.disable(logging.CRITICAL)

        shape = image.shape
        image = YeastMatePredictor.image_to_tensor(image)

        with torch.no_grad():
            instances = self.artifacts.model([image])[0]['instances']

        things, mask = YeastMatePredictor.postprocess_instances(instances, \
                    possible_comps = self.artifacts['model'].metadata['possible_comps'], \
                    optional_object_score_threshold=self.artifacts['model'].metadata['optional_object_score_threshold'], \
                    parent_override_threshold=self.artifacts['model'].metadata['parent_override_threshold'],
                    score_thresholds=annotations)

        mask = Image.fromarray(mask)
        imagebytes = BytesIO()
        mask.save(imagebytes, format="TIFF")
        mask_data = base64.b64encode(imagebytes.getvalue())

        return {"detections": things, "mask": mask_data}
