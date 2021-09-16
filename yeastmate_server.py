import json
import base64
import logging
import os
from io import BytesIO
from http import HTTPStatus
import argparse
import platform

# prevent ugly errors on Windows when we quit with Ctrl-C
# see https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
if platform.system() == 'Windows':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import numpy as np
from flask import Flask, request
from waitress import serve
from tifffile import imread, imsave

from yeastmatedetector import __version__ as yeastmate_version
from yeastmatedetector.inference import YeastMatePredictor


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', action='store_true')
    argparser.add_argument('--gpu_index', type=int, default=0)
    argparser.add_argument('--port', type=int, default=11005)
    argparser.add_argument('--config', type=str, default=os.path.join(os.path.split(__file__)[0], 'yeastmatedetector/configs/yeastmate.yaml'))
    argparser.add_argument('--model', type=str, default=os.path.join(os.path.split(__file__)[0], 'models/yeastmate_weights.pth'))

    args = argparser.parse_args()

    # init app
    app = Flask(__name__)

    # setup predictor with given config, model
    pred = YeastMatePredictor(args.config, args.model)

    # move model to GPU/CPU
    if args.gpu:
        # check if we have a GPU and the specified GPU index is valid
        if torch.cuda.is_available() and torch.cuda.device_count() > args.gpu_index:
            pred.model.to(torch.device(f'cuda:{args.gpu_index}'))
        else:
            logging.warn(f'GPU:{args.gpu_index} is not available, falling back to CPU detection.')
            pred.model.cpu()
    else:
        pred.model.cpu()

    # read postprocessing parameters from config
    cfg_oo_score_thresh =  pred.cfg['POSTPROCESSING']['OPTIONAL_OBJECT_SCORE_THRESHOLD']
    cfg_override_thresh = pred.cfg['POSTPROCESSING']['PARENT_OVERRIDE_THRESHOLD']
    cfg_possible_comps = pred.cfg['POSTPROCESSING']['POSSIBLE_COMPS']

    @app.route('/status', methods=['GET'])
    def get_status():
        # simple endpoint to check if server is running
        # also get some simple stats
        return {
            'name': 'YeastMate',
            'version': yeastmate_version
        }, HTTPStatus.OK

    @app.route('/predict', methods=['POST'])
    def predict():

        # check that we actually received a valid multipart request
        if not ('annotations' in request.files and 'image' in request.files):
            return '', HTTPStatus.BAD_REQUEST

        try:
            # read thresholds from request
            threshs = json.load(request.files['annotations'])

            # read image from request
            img = imread(request.files['image'])
            img = img.astype(np.float32)
            img_t = YeastMatePredictor.image_to_tensor(img)

            # run network
            with torch.no_grad():
                instances = pred.model([img_t])[0]['instances']

            # do postprocessing
            things, mask = YeastMatePredictor.postprocess_instances(instances,
                        possible_comps = cfg_possible_comps,
                        optional_object_score_threshold=cfg_oo_score_thresh,
                        parent_override_threshold=cfg_override_thresh,
                        score_thresholds=threshs)

            # encode mask image as base64
            maskbytes = BytesIO()
            imsave(maskbytes, mask)
            mask_data = base64.b64encode(maskbytes.getvalue())

            # box is still tuple of np.float32
            # convert to tuple of standard float so it is JSON-serializeable
            # TODO: fix upstream
            for v in things.values():
                v['box'] = tuple(map(float, v['box']))

        # something went wrong on the server side
        # TODO: more fine-grained error handling?
        except Exception as e:
            logging.exception(e)
            return '', HTTPStatus.INTERNAL_SERVER_ERROR

        # return with 200
        return {"detections": things, "mask": mask_data.decode('utf-8')}, HTTPStatus.OK

    # run server, listen for external connections as well
    # TODO: make port settable
    # app.run(host='0.0.0.0', port=args.port, debug=False)
    print(f'Running YeastMate detection server on port {args.port}. Press Ctrl-C to quit.')

    serve(app, host='0.0.0.0', port=args.port)

    print('Shutting down...')
