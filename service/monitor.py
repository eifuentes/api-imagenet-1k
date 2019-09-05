import logging
from time import time

import numpy as np

logger = logging.getLogger('app')


class ImageUrl:
    def __init__(self, url, processing_time):
        self.url = url
        self.history = [processing_time]
        self.time = time()

    def beat(self, processing_time):
        self.history.append(processing_time)
        self.time = time()

    def analyze(self):
        nd_history = np.array(self.history)
        min_hist = np.min(nd_history).round(3)
        max_hist = np.max(nd_history).round(3)
        mean_hist = np.mean(nd_history).round(3)
        return {
            'num_requested': nd_history.shape[0],
            'processing_time_min': min_hist,
            'processing_time_max': max_hist,
            'processing_time_mean': mean_hist
        }

    def is_expired(self, maxdur):
        delta = time() - self.time
        return delta > maxdur

    def __len__(self):
        return len(self.history)


class BasicEvaluatorMonitor:
    def __init__(self, maxdur=259200):
        self.maxdur = maxdur
        self.mapping = dict()

    def __call__(self, url, processing_time):
        if url in self.mapping:
            self.mapping[url].beat(processing_time)
        else:
            self.mapping[url] = ImageUrl(url, processing_time)

    def report(self, top=10):
        sorted_urls = sorted(list(self.mapping.keys()),
                             key=lambda url: len(self.mapping[url]),
                             reverse=True)
        top_urls = sorted_urls[:top]

        if len(top_urls) > 0:
            analysis = {url: self.mapping[url].analyze() for url in top_urls}
        else:
            analysis = {'message': f'service has no classify-image requests within {self.maxdur}s'}
        return analysis

    def clear(self):
        stale_urls = list()
        for url, el in self.mapping.items():
            if el.is_expired(self.maxdur):
                stale_urls.append(url)
        for url in stale_urls:
            del self.mapping[url]
