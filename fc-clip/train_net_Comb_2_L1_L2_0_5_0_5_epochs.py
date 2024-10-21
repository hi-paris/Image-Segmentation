import time
import copy
import itertools
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.utils.events import EventStorage
import torch
import detectron2.utils.comm as comm

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)

from fcclip import (
    MaskFormerPanopticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_fcclip_config
)


class CombiningApproachTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Initialize the trainer by building two models: `model` and `model_two`.
        """
        super().__init__(cfg)
        self.model_two = self.build_model_two(cfg)  # Build the second model

    @classmethod
    def build_model_two(cls, cfg):
        """
        Build the second model based on a copy of the configuration.
        """
        model_two_cfg = copy.deepcopy(cfg)
        model_two_cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS_One  # Second model weights
        model = cls.build_model(model_two_cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS_One)  # Load second model weights
        return model

    def run_step(self):
        """
        Implementation of the training step for combining two models.
        """
        assert self.model.training, "[CombiningApproachTrainer] model was changed to eval mode!"
        assert self.model_two.training, "[CombiningApproachTrainer] model_two was changed to eval mode!"

        # Load the data
        data = next(self._trainer._data_loader_iter)

        # Forward pass for both models
        loss_dict_model_one = self.model(data)
        loss_dict_model_two = self.model_two(data)

        # Combine the losses from both models
        alpha1 = 0.5  # weight for model one
        alpha2 = 0.5  # weight for model two

        total_loss = alpha1 * sum(loss_dict_model_one.values()) + alpha2 * sum(loss_dict_model_two.values())

        # Backpropagation
        total_loss.backward()
        self._trainer.after_backward()

        self._trainer.optimizer.step()
        self._trainer.optimizer.zero_grad()

    def train(self, start_iter: int = 0, max_iter: int = None):
        self.iter = start_iter
        self.max_iter = max_iter if max_iter is not None else self.cfg.SOLVER.MAX_ITER
        
        self.scheduler = self.build_lr_scheduler(self.cfg, self.optimizer)
        self.checkpointer = DetectionCheckpointer(
            self.model, self.cfg.OUTPUT_DIR, optimizer=self.optimizer, scheduler=self.scheduler
        )

        # Initialize hooks before starting training
        self.before_train()

        total_epochs = 3000  # Total epochs
        iters_per_epoch = self.cfg.SOLVER.MAX_ITER // total_epochs

        with EventStorage(start_iter) as storage:
            while self.iter < self.max_iter:
                epoch = self.iter // iters_per_epoch

                # Dynamically adjust eval period
                self.cfg.defrost()
                if 0 <= epoch <= 100:
                    self.cfg.TEST.EVAL_PERIOD = 10
                elif 100 < epoch <= 500:
                    self.cfg.TEST.EVAL_PERIOD = 50
                elif 500 < epoch <= 3000:
                    self.cfg.TEST.EVAL_PERIOD = 100
                self.cfg.freeze()

                # Perform a training step
                self.run_step()

                # Evaluate periodically
                if (self.iter % self.cfg.TEST.EVAL_PERIOD == 0) or (self.iter == self.max_iter):
                    self.test(self.cfg, self.model)

                self.iter += 1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(f"No Evaluator for dataset {dataset_name} with type {evaluator_type}")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                params.append({"params": [value], "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY})

        return maybe_add_gradient_clipping(cfg, torch.optim.AdamW(params))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg) 
    cfg.INPUT.CROP.MINIMUM_INST_AREA = 1
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="fcclip")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = CombiningApproachTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = CombiningApproachTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Use the combined trainer with two models
    trainer = CombiningApproachTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args) 
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    torch.cuda.empty_cache()
