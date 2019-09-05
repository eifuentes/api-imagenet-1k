import logging
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger('app')


def _fetch_image(url):
    logger.info(f'fetching image url: {url}')
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)) if resp.ok else None
    return img


def _handle_alpha(img):
    img_with_alpha = img.convert('RGBA')
    background = Image.new('RGBA', img_with_alpha.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, img_with_alpha)
    cleaned_img = alpha_composite.convert('RGB')
    logger.info('handeled image alpha channel')
    return cleaned_img


def fetch_image(url):
    img = _fetch_image(url)
    if img and img.mode == 'RGBA':
        img = _handle_alpha(img)
    return img
