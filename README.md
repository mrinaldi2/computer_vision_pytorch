# computer_vision_pytorch
Classify and predict 102 flowers categories with pytorch and transfer learning


## How to train your model

In order to train the model you need first of all to download the files on your pc.

Once you have them is reccomanded that you put in a folder with your images. Try to avoid paths with space or dotted lines 
you can usually hand up in problems.

In order to train you need the images divided in the following structure. 
Let's say we have dogs and cats:

    pets/
      train/
        0/
        1/
      test/ 
        0/
        1/
      validation/
        0/  
        1/
        
With the folder structured like above we can simply run the train.py script in the following way

```
python train.py --data-dir=pets/
```
Also in order to have the right labels you should upload a file named ```cat_to_name.json``` which contains the following:

```
{"0": "dogs", "1", "cats"} 
```

in order to run the script you need to install torch and torchvision

```
pip install torch
pip install torchvision

```

You can see all the arguments with

```
python train.py --help
usage: train.py [-h] [--data_dir DATA_DIR] [--model_name MODEL_NAME]
                [--save_dir SAVE_DIR] [--learning_rate LEARNING_RATE]
                [--hidden_input HIDDEN_INPUT] [--epochs EPOCHS] [--gpu MODE]

This script helps in training the model

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory of images
  --model_name MODEL_NAME
                        Pretrained model
  --save_dir SAVE_DIR   Directory for checkpoints
  --learning_rate LEARNING_RATE
                        Learning rate for the neural network
  --hidden_input HIDDEN_INPUT
                        size of the hidden input
  --epochs EPOCHS       Epochs to run
  --gpu MODE            Gpu or cpu mode. Try to use gpu when possible to speed
                        up the training
```

The supported pretrained models are:

- vgg19
- densenet121
- resnet152
- alexnet

## How to predict 

Once you have trained the model you'll have a checkpoint.pth file that you will use to load it in the predict.py file

In order to make a prediction just run 

```
python predict.py --image-path [path-to-image]
```

if you saved the checkpoint file in another folder then you'll have to add also the path to the checkpoint file as parameter

```
python predict.py --help
usage: predict.py [-h] [--image_path IMAGE_PATH]
                  [--checkpoint_path CHECKPOINT_PATH] [--top_k TOP_K]
                  [--gpu MODE]

This script helps in predicting the model

optional arguments:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
  --checkpoint_path CHECKPOINT_PATH
  --top_k TOP_K
  --gpu MODE
 ```
