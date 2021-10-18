FROM pytorch/pytorch

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS yes

RUN addgroup --system app && adduser --system app
ENV APP_DIR=/home/metric_learning_classification
COPY --chown=app:app . $APP_DIR
WORKDIR $APP_DIR

RUN apt-get -y update 
RUN apt-get install -y vim && \
    apt-get install -y libopencv-dev && \
    apt-get install -y lsof && \
    apt-get install -y tree 

RUN pip install matplotlib && \
    pip install tensorboard && \
    pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r $APP_DIR/docker/requirements.txt
