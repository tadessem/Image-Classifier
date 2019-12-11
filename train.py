import argparse
import torch
import torch.nn.functional as F
from torch import nn,optim
from torchvision import datasets,transforms,models
from collections import OrderedDict
import numpy
import numpy as np


def main():
    
    arguments = parse_argument()  
    
    architect = arguments.architect
    save_dir = arguments.save_dir
    lr = arguments.learning_rate
    epoch = arguments.epochs
    hidden_units = arguments.hidden_units
    gpu = arguments.gpu
    
    
    
    # directory for the data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #transorm and dataloader
    [trainloader, validloader, testloader,train_dataset] = transform_and_load(train_dir, valid_dir, test_dir)
    
    print("dataloader finished")
    #build a model
   
    model = model_builder(architect,hidden_units)
    print("initial model built")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion=nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr)
    T_step = 10 #number of iteration point to print and validate 
    trained_model = train_network(model, trainloader, validloader, device, criterion, optimizer,epoch, T_step)
    print('Training Finished')
    
    #validate model
    validate_network (trained_model,criterion,testloader,device)
    print('validation on the trained model finished')
    #save checkpoint
    save_checkpoint(trained_model, save_dir, optimizer, train_dataset)
    print('checkpoint saved')
    
def parse_argument():
    # set to get argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--architect', type=str,default = 'densenet169', help = "Pick a pretrained model from 'densenet169','vgg13','alexnet','resnet18'")
    parser.add_argument('--save_dir', type=str, help = 'file name to save checkpoint')
    parser.add_argument('--learning_rate', default = 0.005, type = float, help = 'learning rate to be used during training')
    parser.add_argument('--epochs', default = 2, type = int, help = 'number of epoch to be used to train the model')
    parser.add_argument('--hidden_units', default = 512,type = int, help = 'number of nodes in the hidden layer for the model classifier')
    parser.add_argument('--gpu', action = 'store_true', help='Use GPU + Cuda for calculations')
    
    print('parsing argument finished')
    return parser.parse_args()

def transform_and_load(train_dir,valid_dir, test_dir):
    # transorm the data to make it suitable for training and validation
    train_transforms=transforms.Compose([transforms.RandomRotation(20),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    validation_transforms=transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    test_transforms=transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    # load the data set with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    test_dataset = datasets.ImageFolder(test_dir,transform=test_transforms)
    
    # define the data loader
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_dataset,batch_size=32)
    testloader =  torch.utils.data.DataLoader(test_dataset,batch_size=32)
    
    return trainloader, validloader, testloader, train_dataset

def model_builder (architect ='densenet169',hidden_units = 512):
    
    # build a model using using pretrained network

    if architect == 'densenet169': #setting up densenet169 model
        model = models.densenet169 (pretrained = True)
    if architect == 'vgg13': # seeting vgg13 model
        model = models.vgg13 (pretrained = True)
    if architect == 'alexnet': # setting alexnet model
        model = models.alexnet (pretrained = True)
    if architect == 'resnet18': # setting up resnet18 model
        model = models.resnet18 (pretrained = True)   
        
    model.name = architect
    
    for par in model.parameters():
        par.requires_grad=False   # freeze all the model parameters from backpropagation    
    in_put = model.classifier.in_features
    
    # define classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_put, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))]))  
    model.classifier = classifier
    return model

def train_network(model, trainloader, validloader, device, criterion, optimizer,epoch, T_step):
    model.to(device)
    for e in range(epoch):
        counter = 0
        training_loss = 0
        for img,labels in trainloader:
            counter+=1
            img,labels = img.to(device),labels.to(device) # to load to GPU
            optimizer.zero_grad() # to avoid batch gradient accumulation
        
            output = model.forward(img)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
        
            training_loss+=loss.item()
        
            if counter % T_step == 0:
            
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad(): # turn off gradient 
                    for img,labels in validloader:
                        img,labels = img.to(device),labels.to(device)
                        output = model.forward(img)
                        v_loss=criterion(output,labels)
                        valid_loss+=v_loss.item()

                        ps = torch.exp(output)
                        top_p,top_class = ps.topk(1,dim=1)
                        equals = (top_class == labels.view(*top_class.shape))
                        accuracy+=torch.mean(equals.type(torch.FloatTensor))


                print(f"Epoch:{e+1}/{epoch} and batch: {counter}:",f"Training Loss:{training_loss/T_step:.4f}",
                 f"Validation Loss:{valid_loss/len(validloader):.4f}",
                 f"Test Accuracy:{accuracy/len(validloader):.4f}")
                model.train()
                training_loss = 0
    return model

def validate_network (model,criterion,testloader,device):
    Test_loss = 0
    accuracy = 0
    with torch.no_grad(): # turn off gradient 
        model.eval()
        for img,labels in testloader:
            img,labels = img.to(device),labels.to(device)
            output = model.forward(img)
            T_loss=criterion(output,labels)
            Test_loss+=T_loss.item()

            ps = torch.exp(output)
            top_p,top_class = ps.topk(1,dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy+=torch.mean(equals.type(torch.FloatTensor))
        print(f"Test Loss:{Test_loss/len(testloader):.3f}", f"\n Test Accuracy:{accuracy/len(testloader):.3f}")
        
def save_checkpoint(model, save_dir, optimizer, train_dataset):
    
    ckpt = {'architecture':model.name,
            'classifier':model.classifier,
            'class_to_idx':train_dataset.class_to_idx,    
            'state_dict':model.state_dict(),
            'optimizer':optimizer,
            'optimizer_state_dict':optimizer.state_dict()}
    if save_dir:
        torch.save(ckpt,save_dir)
    else:
        print('checkpoint saved on default checkpoint.path ')
        torch.save(ckpt,'checkpoint.pth')
        
        
# main
if __name__ == "__main__":
    main()

        

        
        

    
    

    
    
    
    
    



    
    
    
    
    
    
    
    
    
    
    
    

    

    
    

    

