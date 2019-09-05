import logging
import os
from timeit import default_timer as timer

import falcon
import torch
from cachetools import TTLCache, cached

from image import fetch_image
from model import SqueezeNetV1Evaluator, SqueezeNetV2Evaluator
from monitor import BasicEvaluatorMonitor

# env vars
CACHE_MAXSIZE = int(os.environ.get('CACHE_MAXSIZE', 1024))
CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))  # default to 1h aka 3600s
MONITOR_WINDOW = int(os.environ.get('MONITOR_WINDOW', 259200))  # default to 3d aka 259200s
REPORT_TOP_N = int(os.environ.get('REPORT_TOP_N', 10))
SQUEEZENET_VERSION = int(os.environ.get('SQUEEZENET_VERSION', 2))

# setup logging
logging_format = '[%(asctime)s] [%(name)s] [%(levelname)s] %(funcName)s:%(lineno)d: %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)

# setup cache and simple monitor for reports
cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
monitor = BasicEvaluatorMonitor(maxdur=MONITOR_WINDOW)

# load pretrained imagenet squeeze net model, load into GPU and optimize if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SqueezeNetEvaluator = SqueezeNetV1Evaluator if SQUEEZENET_VERSION == 1 else SqueezeNetV2Evaluator
evaluator = SqueezeNetEvaluator(device, optimize=True)


@cached(cache)
def _infer(url):
    img = fetch_image(url)
    try:
        pred_class, pred_confidence = evaluator(img)
    except Exception:
        logging.warning('unable to infer classification from image')
        pred_class, pred_confidence = None, None
    return pred_class, pred_confidence


def infer(url):
    start_time = timer()
    pred_class, pred_confidence = _infer(url)
    end_time = timer()
    processing_time = round(end_time - start_time, 5)
    if pred_class:
        monitor(url, processing_time)
    return pred_class, pred_confidence, processing_time


class ClassifyImage:
    def on_post(self, req, resp):
        """Handles POST requests"""
        payload = req.media
        if 'image_url' in payload:  # parse request image_url
            pred_class, pred_confidence, processing_time = infer(url=payload['image_url'])
            if pred_class:  # confirm model inference produced a result
                resp.media = {
                    'classification': pred_class,
                    'confidence': f'{pred_confidence:.3f}',
                    'processing_time': f'{processing_time:.3f}'
                }
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_500
        else:
            resp.status = falcon.HTTP_400


class ServiceReport:
    def on_get(self, req, resp):
        """Handles GET requests"""
        monitor.clear()  # clear out stale urls from report
        analysis = monitor.report(top=REPORT_TOP_N)  # calculate top N url metrics
        if analysis:
            resp.status = falcon.HTTP_200
            resp.media = analysis
        else:
            resp.status = falcon.HTTP_500


api = falcon.API()
api.add_route('/classify-image', ClassifyImage())
api.add_route('/report', ServiceReport())
