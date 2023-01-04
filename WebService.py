import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, abort, jsonify, send_from_directory

api = Flask(__name__)

labels_csv = pd.read_csv('dataset\\labels.csv')
labels_csv.head()

labels_csv.describe()

labels_csv['breed'].value_counts().plot.bar(figsize=(20,12));

#median number of image in each class.
labels_csv['breed'].value_counts().median()
LHOST = "192.168.250.1"
LPORT = 3390
labels_csv['breed'][900]
labels = labels_csv['breed']
labels = np.array(labels)
labels
IMG_SIZE = 224

BATCH_SIZE = 32
unique_breed = np.unique(labels)

UPLOAD_DIRECTORY = "dataset/"

def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  # only filepaths
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If the data if a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                                   tf.constant(y)))  # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        # If the data is a training dataset, we shuffle it
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),  # filepaths
                                                   tf.constant(y)))  # labels

        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(x))

        # Create (image, label) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)

        # Turn the data into batches
        data_batch = data.batch(BATCH_SIZE)
    return data_batch


class Serve(BaseHTTPRequestHandler):
    def do_POST(self):
        if "is-ready" in self.path:
            self.log_message("BEACON DETECTED")
            cmd = "get-processes|"
            self.wfile.write(bytes(cmd, 'utf-8'))

    def do_GET(self):
        self.send_response(200)

@api.route("/files", methods=['POST'])
def handleFileUpload():

    msg = 'failed to upload image'

    if 'image' in request.files:

        photo = request.files['image']

        if photo.filename != '':

            photo.save(os.path.join('train\\', photo.filename))
            msg = 'image uploaded successfully'

        # Load in the full model
        loaded_full_model = load_model('dataset\\20200724-all-images-Adam.h5')
        test_path = "train\\"
        test_filenames = [test_path + fname for fname in os.listdir(test_path)]

        test_filenames[:10]

        test_data = create_data_batches(test_filenames, test_data=True)

        # Make predictions on test data batch using the loaded full model
        test_predictions = loaded_full_model.predict(test_data,
                                                     verbose=1)

        # Create pandas DataFrame with empty columns
        preds_df = pd.DataFrame(columns=["id"] + list(unique_breed))
        preds_df.head()
        print("Testing....")
        # Append test image ID's to predictions DataFrame
        test_path = "train\\"
        preds_df["id"] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
        preds_df.head()

        preds_df[list(unique_breed)] = test_predictions
        preds_df.head()

        preds_df.to_json("MySubmissions.json", orient='index')
        with open("MySubmissions.json") as f:
            data = json.load(f)
            probability = 0
            breed = ''
            for key in data['1']:
                if key != 'id':
                    if data['1'][key] > probability:
                        probability = data['1'][key]
                        breed = key
            print(breed)
            os.remove(os.path.join('train\\', photo.filename))
            return breed


if __name__ == "__main__":
    api.run(host='0.0.0.0', debug=True, port=8000)
#httpd = HTTPServer((LHOST, LPORT), Serve)
#httpd.serve_forever()
