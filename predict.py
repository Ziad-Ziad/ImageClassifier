import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
from nnFunc import load_data, classifier, load_checkpoint, predict

parser = argparse.ArgumentParser()

parser.add_argument('img',  type = str, default='./flowers/valid/11/image_03100.jpg', help='The image path')
parser.add_argument('checkpoint', type = str, default='./checkpoint.pth', help='path to the checkpoint to be loaded')
parser.add_argument('--top_k', type=int, default=5, help='get the top k largest values in a tensor')
parser.add_argument('--category_names', type = str, default='cat_to_name.json', help='The cateogyr of each label')
parser.add_argument('--gpu', action='store_true', help='host the code on the gpu RECOMMENDED')


args = parser.parse_args()

image_path = args.img
checkpoint = args.checkpoint
topk = args.top_k
device1 = args.gpu
categories = args.category_names

if device1:
    device1 = 'cuda'
else: 
    device1 = 'cpu'


def main():
    
    model = load_checkpoint(checkpoint)
    
        
    with open(categories, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(image_path, model, topk)
    
    
#     print(probs)
#     print(classes)
    
    names = []
    for c in classes:
        names.append(cat_to_name[c])
        
        
    res = "\n".join("{}   -->   {}   -->   {}".format(names, classes, probs) for names, classes, probs in zip(names, classes, probs))
    print(res)
    
    
if __name__== "__main__":
    main()
