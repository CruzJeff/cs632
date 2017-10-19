#Script given from https://github.com/random-forests/cs632/blob/master/assignments/cats-vs-dogs/extract_cats_dogs.py
import pickle
import numpy as np
from PIL import Image
import os
import sys

np.random.seed(0)

if not os.path.exists('cifar-10-batches-py'):
  print ("CIFAR-10 data not found. Did you remember to download it?")
  print ("See the comment at the top of this script.")
  sys.exit()

TRAIN_FILES = ['cifar-10-batches-py/data_batch_%d' % i for i in range(1,6)]
TEST_FILE = 'test_batch'

CAT_INPUT_LABEL = 3
DOG_INPUT_LABEL = 5

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo,encoding='latin1') 
    return dict
    
data = []

# Count number of cats/dogs
num_cats = 0
num_dogs = 0

for data_file in TRAIN_FILES:
  d = unpickle(data_file)
  data.append(d)

  for label in d['labels']:
    if label == CAT_INPUT_LABEL:
      num_cats += 1
    if label == DOG_INPUT_LABEL:
      num_dogs += 1
    
total = num_cats + num_dogs
print ("Found %d images" % total)

# Copy the cats/dogs into new array
images = np.empty((num_cats + num_dogs, 32, 32, 3), dtype=np.uint8)
labels = np.empty((num_cats + num_dogs), dtype=np.uint8)
index = 0

for data_batch in data:
  for batch_index, label in enumerate(data_batch['labels']):
    if label == CAT_INPUT_LABEL or label == DOG_INPUT_LABEL:
      # Data is stored in B x 3072 format, convert to B' x 32 x 32 x 3
      images[index, :, :, :] = np.transpose(
          np.reshape(data_batch['data'][batch_index, :],
          newshape=(3, 32, 32)),
          axes=(1, 2, 0))
      if label == CAT_INPUT_LABEL:
        labels[index] = CAT_OUTPUT_LABEL
      else:
        labels[index] = DOG_OUTPUT_LABEL
      index += 1
    
# split the data into train, test 
training_size = int(total * 0.8)
print("Training size", training_size)
print("Testing size", total - training_size)

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

images, labels = unison_shuffled_copies(images, labels)

train_images = images[:training_size]
train_labels = labels[:training_size]

test_images = images[training_size:]
test_labels = labels[training_size:]
    
np.save('train.npy', {'images': train_images, 'labels': train_labels})
np.save('validation.npy', {'images': test_images, 'labels': test_labels})

# Make sure images look correct
print(labels[0])
img  = Image.fromarray(images[0, :, :, :])
img.show()
