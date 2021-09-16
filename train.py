import os
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, DefaultTrainer, launch, default_setup
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetCatalog, MetadataCatalog

from yeastmatedetector.data import MaskDetectionLoader, DictGetter
from yeastmatedetector.utils import copy_code, initialize_new_config_values

from yeastmatedetector.models import *
from yeastmatedetector.multimaskrcnn import *

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MaskDetectionLoader(cfg, True))

def setup(args):
    cfg = get_cfg()
    cfg = initialize_new_config_values(cfg)
    cfg.merge_from_file(args.config_file)

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[0], date_time)

    if comm.get_rank() == 0:
        copy_code(cfg.OUTPUT_DIR)

    dict_getter = DictGetter("yeastmate", train_path=cfg.INPUT_DIR)

    DatasetCatalog.register("yeastmate", dict_getter.get_train_dicts)
    MetadataCatalog.get("yeastmate").thing_classes = ["single_cell", "mating", "budding"]

    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron")
    return cfg

def main(args):
    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load()

    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main,
           num_gpus_per_machine=args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           args=(args, ),
           )