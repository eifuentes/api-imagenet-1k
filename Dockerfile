FROM ubuntu:18.04
# update applications
RUN apt-get update
# install os deps
RUN apt-get install -y git python3 python3-pip zlib1g-dev libjpeg-dev
# upgrade python tooling
RUN python3 -m pip install --upgrade setuptools pip
# install python package dependencies
RUN pip3 install cython
RUN pip3 install numpy pillow-simd
RUN pip3 install --no-binary :all: falcon
RUN pip3 install cachetools requests gunicorn
RUN pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# copy app contents
COPY ./service /opt/app/
# change working directory
WORKDIR /opt/app/
# handle build args with defaults
ARG CACHE_MAXSIZE=1024
ARG CACHE_TTL=3600
ARG MONITOR_WINDOW=259200
ARG REPORT_TOP_N=10
ARG SQUEEZENET_VERSION=2
# set env vars based on build args
ENV CACHE_MAXSIZE $CACHE_MAXSIZE
ENV CACHE_TTL $CACHE_TTL
ENV MONITOR_WINDOW $MONITOR_WINDOW
ENV REPORT_TOP_N $REPORT_TOP_N
ENV SQUEEZENET_VERSION $SQUEEZENET_VERSION
# open default app port
EXPOSE 8000
# run web service
CMD ["gunicorn", "-b", ":8000", "app:api"]
