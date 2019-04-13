# paperiano

Play a piano drawn on paper using Computer Vision and Deep Learning.

## Installation

### Preparation of the image dataset

1. Download videos, and place them under a `./data/videos` directory.
2. Extract frames from the videos:

```bash
python cli.py readframes data/videos/flying.MOV data/images --start 15 --end 145
python cli.py readframes data/videos/touching.MOV data/images --start 0 --end 120
```

## Binary classifier

Inspired from : https://github.com/perseus784/BvS

This folder contains the classifier touching/flying detecting wether a finger is touching the piano or not.

1) In binary_classifier, create a raw_data folder and 2 subfolders corresponding to the images labels and containig the raw images.
2) Define the layers in model_architecture.py
3) Define hyperparameters and I/O paths in config.py. run config it will crate a processed_data folder. Extract images from there to create a val_set and test_set.
4) Train with trainer.py. The code will use processed_data as train set and val_set as validation set. The model will be saved in checkpoints
5) Predict with predict_test_set.py. The code will use test_set to predict the data and save tes results as pickle files.
6) Run error.py to calculate the error rate.

