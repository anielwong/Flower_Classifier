## PROGRAMMER: ANIEL WONG
 # DATE CREATED: 15 august 2018
 #
 # PURPOSE: Train a pretrained network for flower image classification
 #          and save the network's parameters
 #
 # Example call in command line:
 #    python train.py --data_directory 'flowers' --arch alexnet --epochs 4
 ##

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import argparse
import utility_functions as uf

import json

import time
from time import time

def main():
    print('*** Retrieving model and hyperparameters ***')
    
    # Retrieving the arguments passed in the command line
    args = get_input_args()

    # Retrieving images to train, test and validate
    image_datasets, dataloaders = uf.get_dataloaders(args.data_directory, args.batch_size)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    ### CREATION OF OUR MODEL

    #inputs_hidden_layers = args.hidden_units 
    # Creation of a list of string of the values of the hidden layers
    inputs_hidden_layers = args.hidden_units.split(',')

    # Transforming the strings from the command line into integers
    for units in range(len(inputs_hidden_layers)):
        inputs_hidden_layers[units] = int(inputs_hidden_layers[units])

    # Creation of our model according to the architecture chosen in command line
    our_model = uf.create_model(args.arch)
    # Modifying the classifier of our model according to the inputs in command line
    our_model = uf.create_classifier(our_model, inputs_hidden_layers, len(cat_to_name), args.dropout)
    
    # The model that we will use for training
    print('Model created and to be used:\n')
    print(our_model)

    # Choosing the classifier and the optimizer as NLLLoss and Adam respectively
    # We only train the classifier parameters, the features parameters are frozen
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(our_model.classifier.parameters(), lr = args.learning_rate)
    
    # Attribution of the device according to the availability of GPU or CPU and the input in command line
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = args.device

    ### TRAINING THE NETWORK
    
    print('*** STARTING TRAINING ***')
    
    # Training function implementation
    start_time = time()
    steps = 0

    # Using GPU to do calculations for nn
    our_model.to(device)

    # Activation of training mode of the network
    our_model.train()

    for e in range(args.epochs):
        running_loss = 0

        # Looping throught the train laoders
        for inputs, labels in dataloaders['train']:
            steps += 1

            # Settings to GPU or CPU for calculations
            inputs, labels = inputs.to(device), labels.to(device)

            ## FORWARD & BACKPROPAGATION
            optimizer.zero_grad()
            
            output = our_model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

            # When it is time to print
            if steps % args.print_every == 0:

                # We make sure that the network is in eval mode for Inference by turning off autograd
                our_model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(our_model, dataloaders['validation'], criterion, device)

                print("Epoch: {}/{}... ".format(e+1, args.epochs),
                      "Training Loss: {:.4f}".format(running_loss/args.print_every),
                      "Valid Loss: {:.4f}".format(test_loss/len(dataloaders['validation'])),
                      "Valid Accuracy: {:.4f}".format(accuracy/len(dataloaders['validation'])))

                running_loss = 0

                # We make sure that the network is in train mode again, by turning on autograd
                our_model.train()           

    print('*** Network is Trained ***')
    print('\nFinal Training Loss: {:.4f} Final Valid Loss: {:.4f} Final Valid Accuracy: {:.4}'.format(running_loss/args.print_every, test_loss/len(dataloaders['validation']), accuracy/len(dataloaders['validation'])))
    
    # Saving the mapping of classes to indices into the model
    our_model.class_to_idx = image_datasets['train'].class_to_idx
    
    ## Saving the trained network
    uf.save_checkpoint(args.save_dir , our_model, inputs_hidden_layers, len(cat_to_name), args.dropout)
    
    # Time it took to train the model created            
    tot_time_batch = time() - start_time
    hours = int(tot_time_batch/3600)
    minutes = int((tot_time_batch%3600)/60)
    seconds = int((tot_time_batch%360)%60)
    print("\n** Time per batch: {} seconds".format(tot_time_batch/args.epochs))
    print("\n** Total Elapsed Runtime: {}:{}:{}".format(hours,minutes,seconds))
    
def get_input_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_directory', type = str, help = 'Path to data directory (without forward slashes); type: int', required = True)
    
    parser.add_argument('--arch', type = str, default = 'vgg11', help = 'The model to be used for predictions: \'vgg11\', \'alexnet\' or \'densenet161\'; default = \'vgg11\'; type: str')
    
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate of the training network; default = 0.001; type: float')
    
    parser.add_argument('--hidden_units', type = str, default = '4000,1000', help = 'String of units of the hidden layer units; recommended: \'vgg11\': \'4000,1000\', \'alexnet\': \'1000,500\', \'densenet161\': \'950\'; type: str')
    
    parser.add_argument('--epochs', type = int, default = 4, help = 'Number of epochs for the training; default = 4; type: int')

    parser.add_argument('--batch_size', type = int, default = 64, help = 'Size of the batches; default =  64; type: int')
    
    parser.add_argument('--dropout', type = float, default = 0.4, help = 'Probability of each units to be dropped out during training; default = 0.4; type: flaot')
    
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device to use for calculation: \'gpu\' or \'cpu\'; default = \'cpu\'; type: str')

    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'The model to be used for predictions; default = \'cat_to_name.json\'; type: str')
    
    parser.add_argument('--print_every', type = int, default = 20, help = 'Print every value; default = 20; type: int')
    
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Set directory to save checkpoints; default = \'checkpoint.pth\'; type: str')
    
    return(parser.parse_args())

def validation(model, test_loader, criterion, device):
    # Using device for calculation    
    model.to(device)
    # Setting the model in eval mode for less 
    model.eval()
    
    test_loss = 0
    accuracy = 0
        
    # Looping through the validation datas in order to get the loss values
    for images, labels in test_loader:
        
        # Calling GPU or CPU
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with new inputs never seen by the network
        output = model.forward(images)
        
        # TEST LOSS
        # Computation of the loss values
        test_loss += criterion(output, labels).item()
        
        # ACCURACY 
        # The output of the network is a log softmax so we need to 'undo' the log part
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    #print('Test loss/Accuracy: {:.3f}/{:.3f}'.format(test_loss/len(test_loader), accuracy/len(test_loader)))
    return(test_loss, accuracy)

if __name__ == "__main__":
    main()
