## PROGRAMMER: ANIEL WONG
 # DATE CREATED: 15 august 2018
 #
 # PURPOSE: Utility functions to be used for train.py and predict.py
 ##

import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models 
from collections import OrderedDict

import numpy as np
from PIL import Image

def get_dataloaders(data_directory, batch_size):

    # Getting the directory
    train_dir = data_directory  + '/train'
    valid_dir = data_directory  + '/valid'
    test_dir = data_directory  + '/test'
    
    # Transform functions
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.Resize(224),
                                                    transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])]),
                       'validation': transforms.Compose([transforms.Resize(224),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                                              [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                      'test': datasets.ImageFolder(test_dir, transform = data_transforms['test']),
                      'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['validation'])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = batch_size, shuffle = True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = batch_size),
                   'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size = batch_size)}
    
    return(image_datasets, dataloaders)

def process_image(image_path):
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
    # returns an Numpy array
    
    # Returns an Image object
    image_pil = Image.open(image_path)
    
    # Ratio aspect of the image
    size_ratio = float(image_pil.size[0])/float(image_pil.size[1])
        
    # Resizing the image while keeping the ratio aspect
    # size_ratio < 1: horizontal picture
    if size_ratio < 1:
        resized_image = image_pil.resize((256,int(256/size_ratio)))
        
    # size_ratio >1: vertical picture
    elif size_ratio > 1:
        resized_image = image_pil.resize((int(256*size_ratio), 256))
        
    # size_ratio == 1: square picture
    elif size_ratio == 1:
        resized_image = image_pil.resize((256,256))
    
    # The tuple below (a,b,c,d) means a, b, c and d are for the 
    # left, top, right and lower side of rectangle respectively
    image_pil = resized_image.crop((16,16,256, 256))
    
    # Converts the color channels into 0-1 instead of 0-255
    np_image = np.array(image_pil)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # We want to substract the mean from each color channel and divide by the standard deviation
    norm_image_pil = (np_image - mean)/std
    
    # Changing the dimension of color channel to be 1st and the other unchanged
    final_image_pil = norm_image_pil.transpose((2,0,1))
    
    return(final_image_pil)

def save_checkpoint(name_checkpoint, model, hidden_layers, output, dropout):
    # Saving the model under the name of name_checkpoint with its components

    # Creation of the dictionary with important elements that will be used during the loading of the checkpoint
    checkpoint = {'state_dict': model.state_dict(),
                  'mapping_train': model.class_to_idx,
                  'ordered_dict': model.ordered_dict,
                  'model_name': model.name,
                  'hidden_layers': hidden_layers,
                  'output_size': output,
                  'dropout': dropout}

    # Saving the checkpoint
    torch.save(checkpoint, name_checkpoint)

def load_checkpoint(filepath):
    # Loads the checkpoint

    # Loading the saved checkpoint
    checkpoint = torch.load(filepath)
    
    # Creation of a pretrained model according to the pretrained model used for the checkpoint
    new_model = create_model(checkpoint['model_name'])
    
    # Creating the same classifier as 
    new_model = create_classifier(new_model, checkpoint['hidden_layers'], checkpoint['output_size'], checkpoint['dropout'])
    
    # Modifying the classifier of the pretraied model created above
    new_model.classifier = nn.Sequential(checkpoint['ordered_dict'])
    new_model.class_to_idx = checkpoint['mapping_train']
    
    new_model.load_state_dict(checkpoint['state_dict'])
        
    return(new_model)

def create_model(name_model):
    
    if name_model == 'vgg11':
        new_model = models.vgg11(pretrained = True)
        new_model.name = 'vgg11'

    elif name_model == 'alexnet':
        new_model = models.alexnet(pretrained = True)
        new_model.name = 'alexnet'

    else:
        new_model = models.densenet161(pretrained = True)
        new_model.name = 'densenet161'
        
    return(new_model)

def create_classifier(model_to_modify, hidden_layers, output_size, dropout):
    
    # According to the model to modify, the position of the input feature in the unmodified model is different
    # and its size will change also
    if model_to_modify.name == 'alexnet':
        input_size = model_to_modify.classifier[1].in_features
    elif model_to_modify.name == 'densenet161':
        input_size = model_to_modify.classifier.in_features
    else:
        input_size = model_to_modify.classifier[0].in_features    

    # Creation of a list to be used in the Ordered Dictionary below
    list_inside_ordered_dict = []

    # If any hidden layer is inputted in the command line we create the input layer to the last hidden layer
    if len(hidden_layers)>=1:

        # Input layer to 1st hidden layer
        list_inside_ordered_dict += (('fc1', nn.Linear(input_size, hidden_layers[0])),
                                     ('relu1', nn.ReLU()),
                                     ('dropout1', nn.Dropout(dropout)))

        # 1st hidden layer to the nexts ones
        for hl in range(2, len(hidden_layers)+1, 1):
            list_inside_ordered_dict += (('fc'+str(hl), nn.Linear(hidden_layers[hl-2], hidden_layers[hl-1])), 
                                         ('relu'+str(hl), nn.ReLU()),
                                         ('dropout'+str(hl), nn.Dropout(dropout)))

        # Last hidden layer to the output layer
        list_inside_ordered_dict += (('fc'+str(len(hidden_layers)+1),nn.Linear(hidden_layers[-1], output_size)), 
                                     ('output', nn.LogSoftmax(dim=1)))
    
    # If there is no hidden layer inputted in the command line, there's only the input layer linked to the output layer
    else:
        list_inside_ordered_dict += (('fc1',nn.Linear(input_size, output_size)), 
                                     ('output', nn.LogSoftmax(dim=1)))
    
    ## Creation of the final modified model
    
    # Elements to be saved in the model
    model_to_modify.ordered_dict = OrderedDict(list_inside_ordered_dict)
    model_to_modify.hidden_layers = hidden_layers
    model_to_modify.output_size = output_size
    model_to_modify.dropout = dropout

    # Turning off the autograd in the features' parameters
    for param in model_to_modify.features.parameters():
        param.requires_grad = False
    model_to_modify.classifier = nn.Sequential(model_to_modify.ordered_dict)
    
    # Turning on the autograd in the classifier's parameters    
    for param in model_to_modify.classifier.parameters():
        param.requires_grad = True 

    return(model_to_modify)
