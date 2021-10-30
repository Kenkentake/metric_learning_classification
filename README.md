# metric_learning_classification
This is the repository that uses deep metric learning for the CIFAR10 classification task.

## Requirements
I have confirmed that it works with the following versions.
-  CUDA Version: 10.2
-  Driver Version: 430.26

## How to Setup
You can build the environment by running the following commands
#### 0: Clone repo
```
$ git clone git@github.com:svishwa/crowdcount-mcnn.git
```

#### 1: Build docker image
```
$ docker build -t {image name} -f docker/Dockerfile .
```

#### 2: Create docker container
```
$ docker run --rm -dit -p {container port}:{local port} -e TZ=Asia/Tokyo --gpus all --shm-size=16gb -v /home/{user name}/:/home/ --name {container name} {image name}
```

## How to Train feature extractor
You can train two kinds of deep metric learning.
#### triplet loss
```
$ python train.py --cfg_file './config/triplet_net.yaml' --run_name 'triplet_net' --seed 42
```
#### arcface
```
$ python train.py --cfg_file './config/arcface.yaml' --run_name 'arcface' --seed 42
```
## How to Train classifier
And then you can train k-Nearest Neighbor or cosine similarity classifier using the model trained above as the feature extractor. <br>
Firstly, specify the path of the weight file you saved in `config/knn.yaml`
#### triplet loss + kNN
```
$ python train_knn.py --cfg_file './config/knn.yaml' --run_name 'triplet_knn' --seed 42
```
#### arcface + kNN
```
$ python train_knn.py --cfg_file './config/knn.yaml' --run_name 'arcface' --seed 42
```
#### triplet loss + cosine similarity
```
$ python train_knn.py --cfg_file './config/cos_similarity.yaml' --run_name 'triplet_knn' --seed 42
```
#### arcface + cosine similarity
```
$ python train_knn.py --cfg_file './config/cos_similarity.yaml' --run_name 'arcface' --seed 42
```

## How to Visualize the result
You can check the learning curve, the 2D visualization of embedding space and confusion matrix by running the following command
```
$ tensorboard --host 0.0.0.0 --logdir ./lightning_log --port {local_port}
```

