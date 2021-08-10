import argparse
from inference import YeastMatePredictor

from bentoapi import YeastMate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create BentoML container.')

    parser.add_argument('config_path', type=str, help='Path to config file.')
    parser.add_argument('weight_path', type=str, help='Path to model weight.')

    args = parser.parse_args()

    maskpred = YeastMatePredictor(args.config_path, args.weight_path)

    cfg = maskpred.cfg

    svc = YeastMate()

    postproc = {'possible_comps': cfg.POSTPROCESSING.POSSIBLE_COMPS, \
            'optional_object_score_threshold': cfg.POSTPROCESSING.OPTIONAL_OBJECT_SCORE_THRESHOLD, \
            'parent_override_threshold': cfg.POSTPROCESSING.PARENT_OVERRIDE_THRESHOLD}

    svc.pack('model', maskpred.model, metadata=postproc)

    saved_path = svc.save()
    print('Bento Service Saved in ', saved_path)