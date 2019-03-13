# paperiano

Play a piano drawn on paper using Computer Vision and Deep Learning.

## Binary classifier

Inspired from : https://github.com/perseus784/BvS

This folder contains the classifier touching/flying detecting wether a finger is touching the piano or not.

1) In binary_classifier, create a raw_data folder and 2 subfolders corresponding to the images labels and containig the raw images.
2) Define the layers in model_architecture.py
3) Define hyperparameters and I/O paths in config.py
4) Train with trainer.py, the code will automatically preprocess images and write them in the processed_data folder. The model will be saved in checkpoints
5) Predict with predict.py

TODO :: Split train/val/test, Plot some losses through training