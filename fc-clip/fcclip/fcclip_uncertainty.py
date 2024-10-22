from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
import os
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
import pickle
from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits


VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

class UncertaintyDecoderAdapter(nn.Module):
    """
    Decoder adapter that outputs both segmentation map and uncertainty.
    """
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(bottleneck_dim, input_dim, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(bottleneck_dim)  # Use BatchNorm2d instead of LayerNorm
        self.norm2 = nn.BatchNorm2d(input_dim)  # Use BatchNorm2d instead of LayerNorm
        self.relu = nn.ReLU()
        
        # Additional layer to estimate uncertainty
        self.uncertainty_layer = nn.Conv2d(input_dim, 1, kernel_size=1)
        
    def forward(self, x):
        identity = x  # skip connection
        x = self.conv1(x)
        x = self.norm1(x)  # Apply BatchNorm2d
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)  # Apply BatchNorm2d
        x += identity
        x = self.relu(x)
        
        # Predict uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_layer(x))
        
        return x, uncertainty

class MoEGate(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
    
    def forward(self, decoder_outputs, uncertainties):
        weights = torch.softmax(-torch.stack(uncertainties, dim=0), dim=0)
        final_output = sum(w * out for w, out in zip(weights, decoder_outputs))
        return final_output

@META_ARCH_REGISTRY.register()
class FCCLIP(nn.Module):
    @configurable
    def __init__(self, *, backbone: Backbone, sem_seg_head: nn.Module, criterion: nn.Module,
                 num_queries: int, object_mask_threshold: float, overlap_threshold: float,
                 train_metadata, test_metadata, size_divisibility: int,
                 sem_seg_postprocess_before_inference: bool, pixel_mean: Tuple[float],
                 pixel_std: Tuple[float], semantic_on: bool, panoptic_on: bool, instance_on: bool,
                 test_topk_per_image: int, geometric_ensemble_alpha: float,
                 geometric_ensemble_beta: float, ensemble_on_valid_mask: bool):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        # Define decoder adapters and MoE gate
        self.num_experts = 3
        self.decoders = nn.ModuleList([UncertaintyDecoderAdapter(1024, 64) for _ in range(self.num_experts)])
        self.gate = MoEGate(self.num_experts)

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent)

        # Additional text classification and metadata preparation
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',')  # multiple synonyms for single class
                res.append(x_)
            return res

        try:
            class_names = split_labels(metadata.stuff_classes)
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=""):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)
        return category_overlapping_mask, num_templates, templated_class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(self.train_class_names[idx: idx + bs], self.device).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(self.test_class_names[idx: idx + bs], self.device).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        # Add deep supervision settings if applicable
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        
        # Define class, mask, and dice weights for the loss function
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        
        # Instantiate the Hungarian matcher
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        # Create a dictionary for the weights
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        
        # If deep supervision is enabled, add additional weights for decoder layers
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        losses = ["labels", "masks"]
    
        # Instantiate the SetCriterion object with the required arguments
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK
        }


    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # Extract features from backbone
        features = self.backbone(images.tensor)
        if isinstance(features, dict):
            x = features.get('res4')
        else:
            x = features
        
        decoder_outputs = []
        uncertainties = []
        for decoder in self.decoders:
            output, uncertainty = decoder(x)
            decoder_outputs.append(output)
            uncertainties.append(uncertainty)
        
        final_output = self.gate(decoder_outputs, uncertainties)
        text_classifier, num_templates = self.get_text_classifier()
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates
        outputs = self.sem_seg_head(features)

        if self.training:
            # Prepare targets for training
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # Compute the losses
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            # For inference
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.panoptic_on:
                    panoptic_seg, segments_info = self.panoptic_inference(mask_cls_result, mask_pred_result)
                    panoptic_seg = self.panoptic_seg_postprocess(panoptic_seg, image_size, height, width)
                    processed_results.append({"panoptic_seg": (panoptic_seg, segments_info)})
                elif self.instance_on:
                    instances = self.instance_inference(mask_cls_result, mask_pred_result)
                    instances = detector_postprocess(instances, height, width)
                    processed_results.append({"instances": instances})
                else:
                    sem_seg = self.semantic_inference(mask_cls_result, mask_pred_result)
                    sem_seg = sem_seg_postprocess(sem_seg, image_size, height, width)
                    processed_results.append({"sem_seg": sem_seg})

            return processed_results

    def panoptic_seg_postprocess(self, panoptic_seg, image_size, output_height, output_width):
        """
        Resize the panoptic segmentation prediction to the desired size.
        """
        panoptic_seg = panoptic_seg.unsqueeze(0).float()
        panoptic_seg = F.interpolate(panoptic_seg.unsqueeze(0), size=(output_height, output_width), mode='nearest')
        panoptic_seg = panoptic_seg.squeeze(0).squeeze(0).to(torch.int32)
        return panoptic_seg

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append({"labels": targets_per_image.gt_classes, "masks": padded_masks})
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0
        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info  # Ensure tuple return
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class)
                    })

            return panoptic_seg, segments_info  # Ensure tuple return

    def instance_inference(self, mask_cls, mask_pred):
        image_size = mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // num_classes

        mask_pred = mask_pred[topk_indices]

        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
            result.pred_masks.flatten(1).sum(1) + 1e-6
        )
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
