import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import torch.nn.functional as F

def load_data(data_dir):
    
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    
    trainloader  = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, valid_data,test_data, trainloader, validloader, testloader


def classifier(arch='vgg19', dropout=0.2, h_layers=4096, lr=0.001, device1='cpu'):
#     print(torch.cuda.is_available())
#     print(device1='cuda')
    if device1 == 'cpu':
        device1 = torch.device("cpu")
        
    elif torch.cuda.is_available() and device1 == 'cuda':
        device1 = torch.device("cuda")
    
    else:
        device1 = torch.device("cpu")
        print('CUDA is not available')
        while True: 
            inp = str(input("Type Yes to use the CPU and No to exit... \n"))
            if type(inp) == str:   # if the user presses enter so there will be an error , so instead of makein a try except, we can use this!
                if inp.lower() == 'yes':
                    break
                elif inp.lower() == 'no':
                    exit()

            
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
                                   
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    elif arch == 'inceptionv3':
        model = models.inceptionv3(pretrained=True)
                                   
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
                                   
    else: 
        print(arch, 'Model is not supportted!')
        print('Defult ---> VGG-19')
        model = models.vgg19(pretrained=True)
 
    model.to(device1)

    
    for para in model.parameters():
        para.requires_grad = False

    from collections import OrderedDict    
    

    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features , h_layers),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(h_layers, 102),
                                     nn.LogSoftmax(dim=1))
    
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
        
# def save_checkpoint(model = model, checkp_path = checkp_path, train_data= train_data, input_size=25088, output_size=102, hidden_layers=[4096], h_layers=[4094], arch='vgg19', epochs=3):

    

    
    return model, criterion, device1


def load_checkpoint(f = 'checkpoint.pth'):
    
    checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
    
    if checkpoint['Architecture'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        
    elif checkpoint['Architecture'] == 'alexnet':
        model = models.alexnet(pretrained=True)
                                   
    elif checkpoint['Architecture'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    elif checkpoint['Architecture'] == 'inceptionv3':
        model = models.inceptionv3(pretrained=True)
                                   
    elif checkpoint['Architecture'] == 'resnet18':
        model = models.resnet18(pretrained=True)
                                   
    else: 
        model = models.vgg19(pretrained=True)
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
    


model = load_checkpoint('checkpoint.pth')
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_pil = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    image = image_transforms(image_pil)
    
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()

    image = process_image(image_path)
    
    image = image.resize(1, 3, 224, 224)
    
#     print(image.shape)

    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        
        top_prob = top_prob.exp()  # exp(log(x)) = x  I LOVE MATHS
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes
