# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.
I uses pretrained densenet169 from torchvision.models. Modified the pretrained model classifier with a new classifier with two fully connected layer with dropout and ReLU activation. The loss and accuracy on the validation set were displayed to track the training phase.


optional arguments to use train.py in command line are the following:

'--architect', default = 'densenet169', Pick one pretrained model from 'densenet169','vgg13','alexnet','resnet18'
'--save_dir', file name to save checkpoint
'--learning_rate', default = 0.005,'learning rate to be used during training'
'--epochs', default = 2,'number of epoch to be used to train the model'
'--hidden_units', default = 512,'number of nodes in the hidden layer for the model classifier'
'--gpu','option to use GPU'

optional arguments to use predict.py in command line are the following:
'--image',default = 'flowers/train/1/image_06747.jpg',an image file for prediction
'--topk',default = 5,'number of top most classes to be displayed'
'--checkpoint', default = 'checkpoint.pth','checkpoint file as str'
'--category_names',default = "cat_to_name.json",'Mapping of categories to real names. JSON file name to be provided
'--gpu','option to use GPU'

    

References:
https://modelzoo.co/model/densely-connected-convolutional-networks-2
https://matplotlib.org/tutorials/introductory/pyplot.html
https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
https://docs.python.org/3/library/argparse.html
https://pymotw.com/3/argparse/



