# Flower_Classifier

## Project Overview
This project will implement an image classification application that will train a deep learning model on a dataset of flower images. Then it use the trained model to classify newly flower images.

This project is based on the [AI Programming with Python Udacity Course](https://eu.udacity.com/course/ai-programming-python-nanodegree--nd089).

## Project Highlights
This project is designed to apply essential concepts of artificial intelligence: the programming tools (Python, NumPy, PyTorch), the math (linear algebra, vectors, matrices), and the key techniques of neural networks (gradient descent and backpropagation).

This project contains several files:

    Image_Classifier_Project.ipynb: This jupyter notebook is where the all the codes are.
    predict.py : This file predicts the name of the flower in the image according to the pretrained CNN parameters. This returns the K's biggest probable names for the flower.
    train.py : This file trains a pretrained network for flower image classification and save the network's parameters.
    utility_functions.py : This file contains functions that are used in the predict.py and train.py files. .

## Software and Libraries
This project uses the following software and Python libraries:

- [Python](https://www.python.org/download/releases/3.0/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)(v0.17)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

The softwares will need to be installed and ready to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If Python is not installed yet, it is highly recommended to install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

## Project Instructions

There are two ways to use this project.

1) Use the jupyter notebook **only**.

2) Use the `train.py`, `predict.py` and `utility_functions.py` with a **command prompt**.

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/anielwong/dog_breed_classifier.git
cd dog_breed_classifier
```

2. Download the [flower dataset](www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz). Unzip the folder and place it in the repo, at location `path/to/Flower_Classifer/flowers`

3. If using the notebook, open the notebook and follow the instruction in it.
```
jupyter notebook Image_Classifier_Project.ipynb
```

4. If using the command prompt, navigate at location `path/to/Flower_Classifier`
```
jupyter notebook dog_breed_classifier.ipynb
```

  a. In the prompt, type
  ```
  python train.py -h
  ```
  to have the options for architectures and values of different parameters.
  
  b. An example call in the prompt:
  ```
  python train.py --data_directory 'flowers' --arch alexnet --epochs 4
  ```  
  
  c. Once all the parameters chosen and the training session is finished, type
  ```
  python predict.py -h
  ```  
  to have all the requirements and options for image prediction.
  
  d. An example call in the prompt
  ```
  python predict.py --checkpoint 'checkpoint_vgg11.pth' --image_to_predict 'flowers/test/100/image_07926.jpg'
  ```
 
