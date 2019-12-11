import argparse
import torch
from torch import nn,optim
from torchvision import datasets,transforms,models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot
import json


def main():
    
    arguments = parse_argument()  
    print('parsing argument finished')
    
    image_path = arguments.image
    topk = arguments.topk
    checkpoint = arguments.checkpoint
    gpu = arguments.gpu
    category_names = arguments.category_names
    
    # load the trained model and optimizer
    [model,optimizer] = load_checkpoint(checkpoint) 
    print('checkpoint loaded')
    
    #process image
    processed_image = process_image(image_path)
    print('image processed')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            
    
    [prob, top_flowers] = predict(image_path, model,  device,cat_to_name,topk)
    print('prediction finished')
        
    for i in range (topk):
        print("Number: {} pick from the top {}.. ".format(i+1, topk),"Class name: {}.. ".format(top_flowers [i]),"with               Probability: {:.3f}..%         ".format(prob [i]*100))
        
        
        

def parse_argument():
    # set to get argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default = 'flowers/train/1/image_06747.jpg', help = "image file for prediction")
    parser.add_argument('--topk', type=int, default = 5,help = 'number of top most classes to be displayed')
    parser.add_argument('--checkpoint', default = "checkpoint.pth",type = str, help = 'checkpoint file as str')
    parser.add_argument ('--category_names',default = "cat_to_name.json", help = 'Mapping of categories to real names. JSON file name to be provided cat_to_name.json', type = str)
    parser.add_argument('--gpu', action = 'store_true', help='Use GPU + Cuda for calculations')
    
    return parser.parse_args()

def load_checkpoint(filepath):  # function to load trained model and model parameteres
    checkpoint = torch.load(filepath)
    
    
    if checkpoint['architecture'] == 'densenet169': #setting up densenet169 model
        model_new = models.densenet169 (pretrained = True)
    if checkpoint['architecture'] == 'vgg13': # seeting vgg13 model
        model_new = models.vgg13 (pretrained = True)
    if checkpoint['architecture'] == 'alexnet': # setting alexnet model
        model_new = models.alexnet (pretrained = True)
    if checkpoint['architecture'] == 'resnet18': # setting up resnet18 model
        model_new = models.resnet18 (pretrained = True)   


    
    for par in model_new.parameters():
        par.requires_grad=False   # freeze all the model parameters from backpropagation 
    model_new.class_to_idx = checkpoint['class_to_idx']
    model_new.classifier = checkpoint['classifier']
    model_new.load_state_dict(checkpoint['state_dict'])
    
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model_new,optimizer

def process_image(image): # process a PIL image suitable for pytorch model
    img = Image.open(image)
    width,height = img.size
    
    #find the shortest side
    if width<height:
        n_size = [256,height]
    else:
        n_size = [width,256]
    img.thumbnail(size = n_size)
    n_size = img.size
    
    # find crop parameters and crop to 224x224
    center = [n_size[0]/2,n_size[1]/2]
    img =img.crop((center[0]-244/2,center[1]-244/2, center[0]+244/2,center[1]+244/2))
    
    # convert to 0-1 float
    img = np.array(img)/255
    
    mu = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    img = (img-mu)/std
    
    # change the color channel in the 1st dimension
    img = img.transpose(2,0,1)
    
    return img

def predict(image_path, model, device, cat_to_name, topk=5): # predict image class using the trained model

    #model_p = model.double()
    model_p = model
    model_p.to(device)    
    img_i = process_image(image_path)
    img_i = torch.from_numpy(img_i)
    img_i = img_i.unsqueeze(0)    # to add another dimension to be consistent with array of image input
    
    with torch.no_grad():
        img_i.to(device)
        result = model_p(img_i)
        ps = torch.exp(result)
        top_p,top_class = ps.topk(topk,dim=1)
    clas = np.array(top_class[0])
    prob = np.array(top_p[0])
    indx_to_class = {val: key for key, val in model_p.class_to_idx.items()}

    top_class = [indx_to_class[i] for i in clas]
    top_flowers = [cat_to_name[i] for i in top_class]
    
    
    return prob, top_flowers




# main
if __name__ == "__main__":
    main()



