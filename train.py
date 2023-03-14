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
from nnFunc import load_data, classifier


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, default='../', help='The folder To be trained on')
parser.add_argument('--arch', type=str, default='vgg19', help='CNN model architecture ')
parser.add_argument('--lr', type=float, default=0.001, help='Scaling the gradient\'s step')
parser.add_argument('--epochs', type=int, default=3, help='Number of iterations through the data set')
parser.add_argument('--dropout', type=float, default=0.2, help='The probability of a neuron to be disabled during the training')
parser.add_argument('--hidden_layers', type=int, default=4096, help='Number of Hidden layers')
parser.add_argument('--gpu', action='store_true', help='host the code on the gpu RECOMMENDED')
parser.add_argument('--save_dir', default="./checkpoint.pth", help='Save checkpoint')


args = parser.parse_args()  
data_dir = args.data_dir 
arch = args.arch
lr = args.lr
epochs = args.epochs
dropout = args.dropout
h_layers = args.hidden_layers
device1 = args.gpu
checkp_path = args.save_dir


if device1:
    device1 = 'cuda'
else: 
    device1 = 'cpu'
    
# print(device1)


# if device == 'gpu':  # at first i though we should reight --gpu gpu to use cuda and --gpu cpu to use the cpu 
#     device = 'cuda'

with open('./cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    




# --------------

def main():
    
    train_data, valid_data, test_data, trainloader, validloader, testloader = load_data(data_dir)
    
    # got an error  when wrote device in the args of classifier and assigning the same var to the return "UnboundLocalError: local variable 'device' referenced before assignment"
    model, criterion, device = classifier(arch, dropout, h_layers, lr, device1)
    
    print(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    '''
    def train_nn():

        The func is just to  break from the two loops as using break inside the two loops will not break 
        immediatly but will skip this iteration and move to the second one and i wanna force stop all inner 
        and outer loop so return is the solution!

        uncomment the codes under #EARLY STOPPING and choose when to stop by specifying the epoch and the step 
        each epoch has 103 steps so to quit in the half of the 2nd epoch so epoch=2 and step_stop=

         3 epochs has good accuracy and also the prediction is really good for the imgs so won't train on more than 
         3 epochs to avoid overfitting 
    '''
    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, valid_losses = [], []

    #EARLY STOPPING
    e_stop = None  # each epoch has 103 steps
    step_stop = 30
            
    print("Training The model... ")
    for e in range(epochs):

        #EARLY STOPPING
#             if e == e_stop - 1 and step_stop <= steps:
#                 return
        if e == 0 and step_stop <= steps:
            break
            
        for inputs, labels in trainloader:
            
            if e == 0 and step_stop <= steps:
                break
                
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model = model.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/print_every)
                valid_losses.append(valid_loss/len(validloader)) 

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader)*100:.3f}%")
                running_loss = 0
                model.train()
    print('Finished Training')
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'classifier': model.classifier,
                  'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': h_layers, 
                  'Architecture' : arch,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs,
    #               'learning_rate': 0.001, # lr is included in the optimizer.state_dict() 
                  'state_dict': model.state_dict(),
                  'class_to_idx' : model.class_to_idx}


    torch.save(checkpoint, checkp_path)
    
    print('Checkpoint is saved')
    print('Done')
        
if __name__ == "__main__":
#     app.run_server(debug=True)

    main()

