## PROGRAMMER: ANIEL WONG
 # DATE CREATED: 15 august 2018
 #
 # PURPOSE: Predict the name of the flower in the image according to the pretrained CNN parameters
 #          This returns the K's biggest probable names for the flower.
 #
 # Example call in command line:
 #    python predict.py --checkpoint 'checkpoint_vgg11.pth' --image_to_predict 'flowers/test/100/image_07926.jpg'
 ##

import train
import utility_functions as uf
import torch
from torch import nn
import argparse

import json

def main_predict():
    
    # Retrieving the arguments passed in the command line
    args_predict = get_input_predict_args()
    
    # Attribution of the device according to the availability of GPU or CPU and the input in command line
    if args_predict.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = args_predict.device

    # Loading the checkpoint 
    our_model = uf.load_checkpoint(args_predict.checkpoint)

    with open(args_predict.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    sanity_check(args_predict.image_to_predict, our_model, args_predict.top_k, cat_to_name, device)

def get_input_predict_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type = str, help = 'Checkpoint filepath to be downloaded', required = True)
    
    parser.add_argument('--image_to_predict', type = str, help = 'Pathfile of the image to be predicted', required = True)    
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'The model to be used for predictions; default = \'cat_to_name.json\'; type: str')
    
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top k-th probability to be listed; default = 5; type: int')

    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device to use for calculation: \'gpu\' or \'cpu\'; default = \'cpu\'; type: str')  
    
    return(parser.parse_args())
    
def sanity_check(image_path, model, topk, cat_to_name, device):    
    
    # Processing the image into a readable file for the network model
    ndarray_processed_image = uf.process_image(image_path)
   
    # Computing the k-th highest prediction 
    probs_san_check, classes_san_check = predict(ndarray_processed_image, model, topk, device)

    # Inversing the key-value of the image_datasets because ==> it becomes a 'value-key' dictionary
    # We use this 'value-key' dictionary with the classes' output of predict function
    # Then we use this in order ot use cat_to_name dictionary to get the name of the flower
    value_key = {y:x for x,y in model.class_to_idx.items()}
    flower_names = []
    for i in classes_san_check:
        flower_names.append(cat_to_name[value_key[i]])
    
    print('\nThe most likely class of the flower\'s name is \'{}\' with a probability of {:.4f}\n'.format(flower_names[0].title(),probs_san_check[0]))
    if topk >= 2:
        print('The ', topk-1, ' other possible flower names and their respective probabilities are listed below\n')
        print('FLOWER NAME / PROBABILITY\n')
        for idx in range(1, topk, 1):
            print('{} / {}'.format(flower_names[idx].title(), probs_san_check[idx]))

def predict(ndarray_processed_image, model, topk, device):
    # Predict the class (or classes) of an image using a trained deep learning model.
    
    model.to(device)
    
    model = model.double()
    model.eval()
    
    # Transfer image to numpy array
    torch_image = torch.from_numpy(ndarray_processed_image)
    # adding dimension
    torch_image = torch_image.unsqueeze_(0)
    # Sending image to GPU or CPU for calculation
    torch_image = torch_image.to(device)
    
    # Turning off autograd
    with torch.no_grad():
        output = model.forward(torch_image)
 
    # Output probability of the network
    ps = torch.exp(output)
    
    # Only the top k-th highest value given
    probs_outputs, classes_outputs = ps.topk(topk)
    probs_outputs, classes_outputs = probs_outputs.cpu().numpy(), classes_outputs.cpu().numpy()

    # The values returned by the topk function along with the probabilities 
    # are the values of this dictionary. You need to find the corresponding keys,
    # which are the actual class idx
    
    return(probs_outputs[0], classes_outputs[0])
    
if __name__ == "__main__":
    main_predict()