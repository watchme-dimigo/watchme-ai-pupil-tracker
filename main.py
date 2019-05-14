from keras.models import Sequential
from keras.layers import (
  Conv2D,
  MaxPooling2D,
  Flatten,
  Dense
)

classifier = Sequential()

# 컨볼루션 레이어
classifier.add(
  Conv2D(32, 3, 3, 
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
  activation='relu',
  units=128
))
classifier.add(Dense(
  activation='sigmoid',
  units=1
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

train_datagon = ImageDataGenerator()

training_set = train_datagon.flow_from_directory(
  'dataset/train',
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
