# lsml2_final_project: Skin Cancer classification using Resnet18


## Introduction

This project is about a skin cancer classification project. In this project, I simply divied it into two parts: Model training part and docker part. 

First, in model training part, I will use [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset to train a Resnet model from scratch. And save the best performance model.

Second, in docker part, I will use a django framework as backend, and design a simple HTML frontend for uploading a skin image. Once the skin image is uploaded, it will pass it to the best performance model for prediction, and finaly return the result to the HTML.

<a href="https://colab.research.google.com/drive/1BQn7YQLfj5yJzhJfofESoc-8IaSr_RkG"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

