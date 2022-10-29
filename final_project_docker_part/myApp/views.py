from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from pathlib import Path
import os
from .models import *
from . import models
from .forms import UploadModelForm
from .models import Photo

import torch
import torchvision
from PIL import Image
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
import os


class custom_resnet(nn.Module):
    def __init__(self, in_channel=3, out_features=7, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.resnet = torchvision.models.resnet18(pretrained=False)
            self.out = nn.Linear(in_features=512, out_features=out_features, bias=True)
        elif backbone == "resnet50":
            self.resnet = torchvision.models.resnet50(pretrained=False)
            self.out = nn.Linear(in_features=2048, out_features=out_features, bias=True)
        
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool,
            nn.Flatten(start_dim=1)
        )


    def forward(self, img):
        x = self.backbone(img)
        return self.out(x)

def img_transfrom(img):
    normalize = Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    transform = Compose(
        [
         RandomResizedCrop(224),
         RandomHorizontalFlip(),
         ToTensor(),
         normalize
        ]
    )
    img = transform(img)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return img

def model_predition(img, model):
    img = img_transfrom(img)
    with torch.no_grad():
        out = model(img)
    return out.argmax(-1).item()

resnet18_model = custom_resnet(backbone="resnet18")
resnet18_model.load_state_dict(torch.load("resnet18_19.pth", map_location=torch.device('cpu')))

def hello(request):
	return HttpResponse("Hello world ! ")

def index(request):
    photos = Photo.objects.all()
    skin_label = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}

    form = UploadModelForm()
    
    if request.method == "POST":
        form = UploadModelForm(request.POST, request.FILES)
        

        filename = str(request.FILES['image'])
        path = './media/image/' + filename 

        if form.is_valid():
            form.save()
            img = Image.open(path)
            predict = model_predition(img, resnet18_model)
            context = {
			        'photos': path,
			        'form': form,
			        'predict': skin_label[predict]
			        }

            return render(request, 'myApp/index.html', context)

    predict = None
    photos = None
    context = {
        'photos': photos,
        'form': form,
        'predict': predict,
    }

    return render(request, 'myApp/index.html', context)