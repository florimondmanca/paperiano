import tensorflow as tf
from binary_classifier.utils import utils
from binary_classifier.model import model_tools
from tensorflow.python.client import device_lib
from binary_classifier.config import *
import pickle

print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = model_tools()
model_folder = 'checkpoints'
session = tf.Session()

# Create a saver object to load the model
saver = tf.train.import_meta_graph(model_folder + '.meta')

# restore the model from our checkpoints folder
saver.restore(session, os.path.join(model_folder))

# Create graph object for getting the same network architecture
graph = tf.get_default_graph()

# Get the last layer of the network by it's name which includes all the previous layers too
network = graph.get_tensor_by_name("add_4:0")

# create placeholders to pass the image and get output labels
im_ph = graph.get_tensor_by_name("Placeholder:0")
label_ph = graph.get_tensor_by_name("Placeholder_1:0")

# Inorder to make the output to be either 0 or 1.
network = tf.nn.sigmoid(network)

# Creating the feed_dict that is required to be fed to calculate y_pred
number_of_images = sum([len(files) for r, d, files in os.walk(test_path)])
result_out = []
labels_out = []
for batch in range(int(number_of_images / batch_size_test)):
    tools_test = utils(data_path=test_path)
    images, labels = tools_test.batch_dispatch(batch_size=batch_size_test)
    print('batch_size_test :', batch_size_test)
    print("test set size :", len(labels))
    feed_dict_test = {im_ph: images, label_ph: labels}
    result = session.run(network, feed_dict=feed_dict_test)
    result_out.append(result)
    labels_out.append(labels)
pickle.dump(obj=result_out, file=open('test_results.pickle', 'wb'))
pickle.dump(obj=labels_out, file=open('test_labels.pickle', 'wb'))
