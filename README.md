# Image Classifier API
API to Classify an Image from 1k Classes from the ImageNet 1k challenge.

Provided an image url this service will predict the predominant object in the image. [Here is a list](http://image-net.org/challenges/LSVRC/2012/browse-synsets) of the 1,000 possible objects to be classified.

## Development

To locally develop, follow the docker instructions below to get quickly up and running.

1. docker build -t classifier-api .
2. docker run -p 8000:8000 classifier-api

### Optional

Optionally set any of the below args during the docker build process to override defaults.

- **CACHE_MAXSIZE** *(default: 1024)*
- **CACHE_TTL** *(default: 3600 aka 1h)*
- **MONITOR_WINDOW** *(default: 259200 aka 3d)*
- **REPORT_TOP_N** *(default: 10)*
- **SQUEEZENET_VERSION** *(default: 2)*

## Model

This service leverages SqueezeNet, a highly optimized computer vision classification model similar to MobileNet.
It is a standard computer vision model provided by the PyTorch Vision package `torchvision` in two flavors. Currently the service model uses pre-trained weights built from the 1,000 class ImageNet dataset. Using `PyTorch` 1.0, we can take advantage of the Just-In-Time options for optimizing our computation graph to achieve production level performance similar to `Tensorflow`.

## API

The Image Classifier API is using `Falcon` a highly optimized Python web API framework with currently two routes, one for model inference and one for usage reporting.

- **POST** `classify-image`
- **GET** `report`

In its current implementation there is no authentication or robust logging to a service like Elastic Search/ELK stack and therefore not ready for production.

Caching is provided via `cachetools`, a great python package that consists of a TTL cache/LRU policy.

### POST /classify-image

#### Request

```bash
curl --request POST --header "Content-Type: application/json" --data '{"image_url":"https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"}' http://localhost:8000/classify-image
```

#### Response

```javascript
{"classification": "golden_retriever", "confidence": "0.864", "processing_time": "0.366"}
```

### GET /report

#### Request

```bash
curl --request GET http://localhost:8000/report
```

#### Response
```javascript
{"https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg": {"num_requested": 6, "processing_time_min": 0.0, "processing_time_max": 0.366, "processing_time_mean": 0.061}, "https://cdn.thewirecutter.com/wp-content/uploads/2018/03/womens-running-shoes-lowres-4796-570x380.jpg": {"num_requested": 2, "processing_time_min": 0.0, "processing_time_max": 0.166, "processing_time_mean": 0.083}, "https://s3.amazonaws.com/gumgum-interviews/ml-engineer/cat.jpg": {"num_requested": 1, "processing_time_min": 0.631, "processing_time_max": 0.631, "processing_time_mean": 0.631}}
```

## Known Issues
- upgrade of torchvision version has made predictions incorrect
