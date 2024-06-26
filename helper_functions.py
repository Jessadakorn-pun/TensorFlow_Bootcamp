import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import image
import zipfile
import os
from sklearn import metrics
import random
import datetime

## Create function for unzip file into current directory
def unzip_data(file_name):
  """
  unzip file_name into current directory

  Arg:
    file_name (str): a file path  to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(file_name, 'r')
  zip_ref.extractall()
  zip_ref.close()


# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")



# Plot the validation and training data separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  epoch = range(len(history.history['loss']))

  ## ploting loss curve
  fig, ax = plt.subplots(1, 2, figsize=(20, 5))
  ax[0].plot(epoch, loss, label='Training Loss')
  ax[0].plot(epoch, val_loss, label='Validation Loss')
  ax[0].set_title('Loss Curve')
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  ax[0].legend()

  ax[1].plot(epoch, acc, label='Training Accuracy')
  ax[1].plot(epoch, val_acc, label='Validation Accuracy')
  ax[1].set_title('Accuracy Curve')
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  ax[1].legend()


## Create fuction for evaluate model with testing data
def evaluate_model(model, test_data, categorical=True):
  """
  Returns classification report and confusion metric image.

  Args:
    model: Tensorflow-trained model
    test_data: data object for evaluating
    categorical: True if the problem is categorical classification, False if the problem is binary classification
  """ 

  prediction = model.predict(test_data)

  if categorical:
    prediction_lebel = tf.argmax(prediction, axis=1)
  else:
    prediction_lebel = prediction.round()

  ## ploting confusion matrix
  cm = metrics.confusion_matrix(test_data.labels, prediction_lebel)
  fig = metrics.ConfusionMatrixDisplay(cm)
  fig.plot()

  ## printing classification report
  print(metrics.classification_report(test_data.labels, prediction_lebel))

## Create function for visualize random class of images
def visualize_images(directory, class_names, num_img=5):
  """
  Returns random classes of images

  Args:
    directory (str): the images directory that we want to visualize
    test_data (list): list of classes label
    num_img (int): number of images to be visualize
  """ 
  ## random target class name
  class_name = random.choice(class_names)
  img_dir = directory + '/' + class_name
  ## list all of image's name in target directory
  all_img_name = os.listdir(img_dir)
  ## random num_img name
  img_names = random.sample(all_img_name, num_img)
  
  ## visualize the images
  fig, ax = plt.subplots(1, num_img, figsize=(20, 20))
  for i, img_name in enumerate(img_names):
    ## read in image
    img = image.imread(img_dir + '/' + img_name)
    ax[i].imshow(img)
    ax[i].axis(False)
    ax[i].set_title(class_name)
    

def show_rand_img_test(directory, class_names, num_img=5):
  """
   Returns random images

  Args:
    directory (str): the images directory that we want to visualize
    test_data (list): list of classes label
    num_img (int): number of images to be visualize
  """ 
  fig, ax = plt.subplots(1, num_img, figsize=(20, 20))
  for i in range(num_img):
    class_name = random.choice(class_names)
    target_dir = directory + "/" + class_name
    img_name_all = os.listdir(target_dir)
    img_name = random.sample(img_name_all, 1)

    img = image.imread(target_dir + '/' + img_name[0])
    ax[i].imshow(img)
    ax[i].set_title(class_name)
    ax[i].axis(False)

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  ## read in image
  img = tf.io.read_file(filename)
  ## decode images to tensor
  img = tf.image.decode_jpeg(img)
  ## resize imgage
  img = tf.image.resize(img, [img_shape, img_shape])

  if scale:
    return img / 255.0
  else:
    return img

# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  ## preprossing image
  img = load_and_prep_image(filename)
  ## make prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  if pred[0] > 1:## muticlass_classification
    prob = tf.reduce_max(pred)
    prediction = class_names[int(tf.argmax(pred, axis=1)[0])]
  else:
    prob = pred[0][0]
    prediction = class_names[int(tf.round(pred))]

  ## visualize the result
  plt.imshow(img)
  plt.title(f'Prediction : {prediction}, with confident = {prob * 100:.2f}%')
  plt.axis(False);

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + '/' + experiment_name
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


## Create function for comapre feature extraction and fine tune loss curve
def compare_loss(org_hist, new_hist, initial_epochs=5):
  """
  Compare two TensorFlow history object
  """
  ## Get the original history messure ment
  org_acc = org_hist.history['accuracy']
  org_val_acc = org_hist.history['val_accuracy']

  org_loss = org_hist.history['loss']
  org_val_loss = org_hist.history['val_loss']

  ## Get the new hisrory object
  total_acc = org_acc + new_hist.history['accuracy']
  total_val_acc = org_val_acc + new_hist.history['val_accuracy']

  total_loss = org_loss + new_hist.history['loss']
  total_val_loss = org_val_loss + new_hist.history['val_loss']

  ## plotting loss curve
  fig, ax = plt.subplots(1, 2, figsize=(20, 5))

  ## loss curve
  ax[0].plot(total_loss, label='Training Loss')
  ax[0].plot(total_val_loss, label='Validation Loss')
  ax[0].plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-tuning')
  ax[0].set_title('Loss Curve')
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  ax[0].legend()

  ax[1].plot(total_acc, label='Training Accuracy')
  ax[1].plot(total_val_acc, label='Validation Accuracy')
  ax[1].plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-tuning')
  ax[1].set_title('Accuracy Curve')
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  ax[1].legend()



# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
