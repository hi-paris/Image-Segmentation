"""
Gaëtan Brison 2024
Adapting the forward method to integrate mask predictions into the mask pooling process, especially for out-of-vocabulary cases,
involves modifying the inference part of the forward method.
Below is a full example of how the forward method can be adapted:
"""



def forward(self, batched_inputs):
    images = [x["image"].to(self.device) for x in batched_inputs]
    images = [(x - self.pixel_mean) / self.pixel_std for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)

    features = self.backbone(images.tensor)
    text_classifier, num_templates = self.get_text_classifier()
    text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
    features['text_classifier'] = text_classifier
    features['num_templates'] = num_templates
    outputs = self.sem_seg_head(features)

    if self.training:
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances, images)
        losses = self.criterion(outputs, targets)
        return losses
    else:
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # New ensemble method for mask predictions
        mask_cls_results = self.ensemble_logits_using_mask_prediction(
            features, mask_cls_results, mask_pred_results, text_classifier, num_templates
        )

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        # ... remainder of the inference process as it originally was
        # ...

        return processed_results

def ensemble_logits_using_mask_prediction(self, features, mask_cls_results, mask_pred_results, text_classifier, num_templates):
    clip_feature = features["clip_vis_dense"]
    mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:], mode='bilinear', align_corners=False)
    pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
    
    if "convnext" in self.backbone.model_name.lower():
        pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
    elif "rn" in self.backbone.model_name.lower():
        pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, mask_for_pooling)
    else:
        raise NotImplementedError

    out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)
    in_vocab_cls_results = mask_cls_results[..., :-1] # remove void
    out_vocab_cls_results = out_vocab_cls_results[..., :-1] # remove void

    # Ensemble the in-vocab and out-of-vocab logits
    valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
    valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
    alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
    beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
    alpha = alpha * valid_masking
    beta = beta * valid_masking

    combined_logits = torch.log(
        ((in_vocab_cls_results.softmax(-1) ** (1 - alpha)) * (out_vocab_cls_results.softmax(-1) ** alpha)) +
        ((in_vocab_cls_results.softmax(-1) ** (1 - beta)) * (out_vocab_cls_results.softmax(-1) ** beta))
    )

    mask_cls_probs = torch.cat([combined_logits.softmax(-1), mask_cls_results[..., -1:]], dim=-1)
    mask_cls_results = torch.log(mask_cls_probs + 1e-8)

    return mask_cls_results
