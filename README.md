# lsml2_final_project: Skin Cancer classification using Resnet


## Introduction

This project is about a skin cancer classification project. In this project, I simply divied it into two parts: Model training part and docker part. 

First, in model training part, I will use [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset to train a Resnet model from scratch. And save the best performance model.

Second, in docker part, I will use a django framework as backend, and design a simple HTML frontend for uploading a skin image. Once the skin image is uploaded, it will pass it to the best performance model for prediction, and finaly return the result to the HTML.

<div align="center">
    <a href="./">
        <img src="./images/intro.gif" width="79%"/>
    </a>
</div>

## Run instructions

### Quick start

1. Downlaod or clone this repo
``` shell
cd final_project_docker_part
```

2. docker-compose
``` shell
docker-compose up
```
After runing docker-compose up, it will start the local django server. And then open 127.0.0.1:8000

## Resnet model training part

<a href="https://colab.research.google.com/drive/1BQn7YQLfj5yJzhJfofESoc-8IaSr_RkG"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

In model training part, I use colab to train my model, you can use my colab notebook to reproduce the results.

### Dataset

In this project, I use [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset to train a Resnet model from scratch.

<div align="center">
	<img src="https://i.imgur.com/LYVdBE0.png" title="source: imgur.com" width="80%" />
</div>

The dataset (6GB) contains 10015 images and 1 ground truth response CSV file. 

Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: 
- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- basal cell carcinoma (bcc)
- benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
- dermatofibroma (df)
- melanoma (mel)
- melanocytic nevi (nv)
- vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

For preparing the dataset, split them, 60% goes to training set, 20% goes to validating set and the rest goes to the testing set.

For passing data to dataloader easily, I moved X_train into dataset/train, X_val to dataset/val and X_test to dataset/test, and for each class, make a folder name with class name and store the corresponding images. Just like the figure shows below.

<div align="center">
	<img src="https://i.imgur.com/VfVyZsx.png" title="source: imgur.com" width="40%"/>
</div>

Furthermore, I also uploaded this dataset to my google drive, you can also use this [link](https://drive.google.com/drive/folders/1U1jRNoDF1-__qIWq1Q0tHjuOXUrXF0VV?usp=sharing) to access this dataset, and create a shortcut to your google drvie.

### Architecture, losses, metrics

The model Architecture, I select two models as candidantes. Resnet18 and Resnet50, neither of them don't use pretrain. I train the model from scratch.

For the losses part, I noticed that the dataset is imbalance, hence, I will use weighted cross entropy as the loss function.

<div align="center">
    <a href="./">
        <img src="./images/data_distri.png" width="79%"/>
    </a>
</div>

And for the metrics, I will use F1-score as the main metric to evulate the model performance because of imbalance dataset.

