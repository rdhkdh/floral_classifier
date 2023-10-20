# Floral Classifier  
> A Machine Learning project by Ridhiman Dhindsa

## Description
This is a command-line application developed in Python, which can be used to predict names of flowers from their images. The model has been trained on a dataset consisting of 102 flower categories, divided into training, validation and testing datasets. The directory `flowers/test/` in this repository, contains only a subset of the actual testing dataset. These flower categories can be viewed in the file `cat_to_name.json`. Deep Learning frameworks from PyTorch, such as DenseNet-121 and DenseNet-169, have been used for building the model. The user can use the images provided in the `flowers/test/` directory for prediction, or upload their own image in the project directory and run the prediction script `predict.py`. 

## Requirements
The file `requirements.txt` may be used to create an environment using:  
`conda create --name <env> --file <requirements.txt>`  
**Platform:** win-64  
**System requirements:** Must have Python3 installed, through the Anaconda distribution. For running the prediction script on GPU, the system must have an NVIDIA GeForce RTX GPU as well as the NVIDIA CUDA-Toolkit must be installed.
> If a GPU is not available, the prediction script can be run on the system CPU as well. The command-line application will automatically run on CPU.

## How to Run
1. Clone the repository on your local machine and open a terminal in the project directory.
2. Type the following and press Enter: `python -W ignore predict.py` 
3. Alternatively, you can also set optional arguments via the command line, i.e:  
* `python predict.py <path_to_image> <path_to_checkpoint>`   
Run the prediction script on image specified by path, and using checkpoint file specified by path.
* `python predict.py --top_k <k>`  
Return top k most likely classes. For eg, `python predict.py --top_k 3` : returns top-3 most likely 
classes for the image.
* `python predict.py --category_names <category_to_name.json>`  
Use a particular mapping of categories to real names.
* `python predict.py --gpu`  
Use GPU for inference.
> Simply running `python predict.py` will set the above command line arguments to their defaults:  
path_to_image = `flowers/test/1/image_06743.jpg`  
path_to_checkpoint = `checkpoint.pth`  
top_k = 1  
category_names = `cat_to_name.json`  
gpu = not enabled  



## Tech Stack
Python, PyTorch (cuda, nn, optim modules), TorchVision 

## Training the model
The user can train their own model as well if they wish to. For this, a system with GPU will be necessary.
To get access to the dataset and train your own model, use the following Google Drive link and request access:  
**Flowers Dataset:** [Google Drive Link](https://drive.google.com/file/d/1t5GSDLMkNZkoc9hdUvacuyDTEzg4FPYY/view?usp=share_link)  

**Steps to run the training script:**    
1. Download the Flowers dataset in the directory "flowers". If you ware saving the dataset in a different directory, it must be specified in the command line arguments.    
2. Enable your system GPU and run the following command in terminal: `python train.py --gpu`.  
3. Alternatively, you can also set optional arguments via the command line, i.e:  
* `python train.py <data_dir>`  
Dataset shall be inferred from the 'data_dir' directory.  
* `python train.py --save_dir <save_directory>`  
Set the directory to save checkpoints.  
* `python train.py --arch "densenet121"`  
Choose which model architecture to train on. Models available: densenet121, densenet169.   
*Set Hyperparameters:*       
* `python train.py --learning_rate <0.01>`  
Set the learning rate for the algortihm.  
* `python train.py --hidden_units [500, 256]` 
Set the number of units in the hidden layers. Choose 2 integers between 1024 and 102 in decreasing order.  
* `python train.py --epochs <2>`  
Set number of training epochs.  
* `python train.py --gpu`  
Enable GPU (recommended). 
> Defaults:  
data_dir = `flowers`  
save_dir = root  
arch = densenet121  
learning_rate = 0.01  
hidden_units = [500, 256]  
epochs = 2  
gpu = not enabled  
4. This will create the file `checkpoint.pth` in the desired directory. Now you can run the prediction script on the model you have trained.    

## Further use
This command-line application can be used to train on a variety of other datasets too! Just download a dataset of your choice, along with the category-to-name mapping in a JSON file. Then choose the appropriate command-line arguments and train the PyTorch model on your dataset. The prediction script can then be run on your test images, and can even be integrated with a user interface such as an **app** or **website**, to make predictions.