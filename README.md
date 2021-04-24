# deepFM


# Overview
PyTorch implementation of the paper 
- Huifeng Guo et al.,DeepFM: A Factorization-Machine based Neural Network for CTR Prediction [[link](https://arxiv.org/pdf/1703.04247.pdf)]

# Requirments
This code runs on Docker container. You need to install followings.
- Docker
- docker-compose

# How to use
start container
```
$ cd Docker
$ docker-compose up -d --build
```

```
$ docker exec -it deepFM_experiment /bin/bash
```

run python script in container

```
workspace$ python main.py
```




