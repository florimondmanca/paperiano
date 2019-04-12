from binary_classifier import preprocessing as ppr
import os

# Parameters
raw_data = 'binary_classifier/raw_data'
data_path = 'processed_data'
val_path = 'val_set'
test_path = 'test_set'

height = 64

width = 64

crop_heigth = 30 # in percent

crop_width = 40 # in percent

if not os.path.exists(data_path):
    print('test')
    ppr.image_processing(raw_data, data_path, height, width, crop_heigth, crop_width)

all_classes = os.listdir(test_path)
print(all_classes)

number_of_classes = len(all_classes)
print(number_of_classes)

color_channels = 3

epochs = 2

batch_size = 10
batch_size_val = 50
batch_size_test = 250

batch_counter = -1

model_save_name = 'checkpoints'
