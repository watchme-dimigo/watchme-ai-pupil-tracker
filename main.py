from keras.models import Sequential
from keras.layers import (
  Convolution2D,
  MaxPooling2D,
  Flatten,
  Dense
)

classifier = Sequential()

# 컨볼루션 레이어
classifier.add(
  Convolution2D(32, 3, 3, 
    input_shape=(64, 64, 3),  
    activation='relu')
)

# 풀링 레이어
classifier.add(
  MaxPooling2D(pool_size=(2, 2))
)

# 플래튼 레이어
classifier.add(Flatten())

# full connection
classifier.add(Dense(
  output_dim=128,
  activation='relu'
))
classifier.add(Dense(
  output_dim=1,
  activation='sigmoid'
))

# compiling
classifier.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=[
    'accuracy'
  ]
)

from keras.preprocessing.image import ImageDataGenerator

train_datagon = ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0/2,
  horizontal_flip=False
)

training_set = train_datagon.flow_from_directory(
  'dataset',
  target_size=(64, 64),
  batch_size=32,
  class_mode='binary'
)

from IPython.display import display
from PIL import Image

classifier.fit_generator(
  training_set,
  steps_per_epoch=100,
  epochs=10,
  validation_data=training_set, # change this after test set is made
  validation_steps=800
)
