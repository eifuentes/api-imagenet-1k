import json
import logging

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as vtransforms
from torchvision.models import squeezenet1_0, squeezenet1_1

RESCALE_SIZE = 256
CROP_SIZE = 224
IMAGENET_CLASS_MAP = 'imagenet_class_index.json'

logger = logging.getLogger('app')


def _fetch_imagenet_class_map():
    """Parse ImageNet Class Index JSON"""
    try:
        with open(IMAGENET_CLASS_MAP, 'r') as f:
            class_map = json.load(f)
        logger.info('successfully loaded imagenet class map')
    except Exception:
        raise(f'unable to retrieve class map from {IMAGENET_CLASS_MAP}')
    class_map = {int(i): str(j[1]) for i, j in class_map.items()}
    return class_map


def _maybe_optimize(model):
    try:
        from torch.jit import trace
        model = trace(model, example_inputs=torch.rand(1, 3, 224, 224))
        logger.info('successfully optimized PyTorch model using JIT tracing')
    except ImportError:
        logger.warning('unable to leverage torch.jit.trace optimizations')
        pass
    return model


class ImageNetEvaluator(nn.Module):
    """Evaluator of ImageNet Classes"""
    def __init__(self, device, optimize=False):
        super().__init__()
        self.device = device
        self.optimize = optimize
        self.transform = vtransforms.Compose([
            vtransforms.Resize(RESCALE_SIZE),
            vtransforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
            vtransforms.ToTensor(),
            vtransforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        model = self._fetch_model()
        self.model = model.to(self.device).eval()
        if self.optimize:
            self.model = _maybe_optimize(model)

        self.class_map = _fetch_imagenet_class_map()

    def _fetch_model(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.transform(x).to(self.device)
        num_dims = len(x.size())
        if num_dims != 3:
            raise ValueError('number dimensions of x must be 3')
        with torch.no_grad():
            pred_tensor = self.model(x.unsqueeze(0))
        pred_logproba = F.log_softmax(pred_tensor, dim=1)
        pred_proba, pred_label = torch.max(pred_logproba.detach().exp(), dim=1)
        pred_proba, pred_label = pred_proba.item(), pred_label.item()
        pred_class = self.class_map[pred_label]
        return pred_class, pred_proba


class SqueezeNetV1Evaluator(ImageNetEvaluator):
    """SqueezeNet V1 Evaluator of ImageNet Classes"""
    def _fetch_model(self):
        model = squeezenet1_0(pretrained=True)
        return model


class SqueezeNetV2Evaluator(ImageNetEvaluator):
    """SqueezeNet V2 Evaluator of ImageNet Classes"""
    def _fetch_model(self):
        model = squeezenet1_1(pretrained=True)
        return model
