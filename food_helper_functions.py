# Creating a function that visualies 3 images from target directory
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import numpy as np
import itertools
import tensorflow as tf

def view_three_images(target_dir, target_class):
    """
    Randomly selects and displays 3 random images
    from `target_class` folder in `target_dir` folder.
    """
    target_path = target_dir + target_class
    file_names = os.listdir(target_path)
    target_images = random.sample(file_names, 3)

    plt.figure(figsize=(10, 3))
    for i, img in enumerate(target_images):
        img_path = target_path + "/" + img
        plt.subplot(1, 3, i+1)
        plt.imshow(mpimg.imread(img_path))
        plt.title(target_class)
        plt.axis("off")

def load_and_process(filename, img_shape=224, scale=True):
  """
  Reads in an image, turns it into a tensor and reshapes to (224, 224, 3).

  Paremerets:
  filename: target image string filename;
  img_shape: int size for resizing. default = 224;
  scale: bool, whether to normalize the image - scale
  the values to range (0,1). default = True;
  """

  img = tf.io.read_file(filename)
  img = tf.image.decode_jpeg(img)
  img = tf.image.resize(img, [img_shape, img_shape])

  if scale:
    return img/225.
  else:
    return img

import tensorflow as tw
def pred_plot(model, filename, class_names, scaling=True):
  """
  Imports an image, makes a prediction on it,
  using the chosen trained model and plots the image with the predicted class
  """
  # make the prediction
  img = load_process(filename, scale=scaling)
  pred = model.predict(tf.expand_dims(img, axis=0))

  if len(pred[0]) > 1: #check whether it is multi-class or binary
    pred_class = class_names[pred.argmax()] # multi-class
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # binary

  # plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)

import datetime
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback to store log files

  filepath: "dir_name/experiment_name/current_datetime/"

  dir_name: target directory
  experiment_name: name of experiment directory
  """

  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")

  return tensorboard_callback

from sklearn.metrics import confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
  """
  Creates a labelled confusion matrix comparing predictions and ground truth labels
  (If classes are passed, matrix will be labeled, otherwise integer class values are used)

  Arguments:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

  Returns confusing matrix comparing y_true and y_pred
  """

  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize
  n_classes = cm.shape[0]
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap = plt.cm.Blues) # how correct is (darker - better)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
      labels = np.range(cm.shape[0])
    
      # set the axes
      ax.set(title = "Confusion matrix",
             xlabel = "Predicted label",
             ylabel = "True label",
             xticks = np.arange(n_classes),
             yrange = np.arange(n_classes),
             xticklabels = labels,
             yticklabels = labels)
    
      threshold = (cm.max() + cm.min()) / 2 # set threshold for different colors
    
      # plot the text on the cells
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
          plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                   horizontalalignment = "center",
                   color = "white" if cm[i, j] > threshold else "black",
                   size = text_size)
        else:
          plt.text(j, i, f"{cm[i, j]}",
                   horizontaalignment = "center",
                   color = "white" if cm[i, j] > threshold else "black",
                   size = text_size)
    
        if savefig:
          fig.savefig("confusion_matrix.png")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results_binary(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
