from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.layers import (
  Conv2D,
  MaxPooling2D,
  Flatten,
  Dense
)
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(
  Conv2D(32, (3, 3),
    input_shape=(64, 64, 3),
    activation='relu')
)

classifier.add(
  Conv2D(64, (2, 2),
    input_shape=(32, 3, 3),  
    activation='relu')
)

classifier.add(
  MaxPooling2D(pool_size=(2, 2))
)

classifier.add(Flatten())

classifier.add(Dense(
  activation='relu',
  units=128,
  kernel_initializer=RandomUniform(minval=0.0, maxval=0.0001)
))
classifier.add(Dense(
  activation='sigmoid',
  units=96
))
classifier.add(Dense(
  activation='sigmoid',
  units=64
))
classifier.add(Dense(
  activation='sigmoid',
  units=16
))
classifier.add(Dense(
  activation='softmax',
  units=5
))

classifier.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=[
    'accuracy'
  ]
)

train_datagon = ImageDataGenerator(
  rescale=1./255,
  width_shift_range=0.05,
  height_shift_range=0.05
)
test_datagon = ImageDataGenerator(rescale=1./255)

training_set = train_datagon.flow_from_directory(
  'dataset/train',
  target_size=(64, 64),
  batch_size=32,
  class_mode='categorical'
)

print(training_set.class_indices)

test_set = test_datagon.flow_from_directory(
  'dataset/test',
  target_size=(64, 64),
  batch_size=32,
  class_mode='categorical'
)

classifier.fit_generator(
  training_set,
  steps_per_epoch=180,
  epochs=10,
  validation_data=test_set,
  validation_steps=500,
  callbacks=[
    ModelCheckpoint('./model.h5', monitor='val_loss', save_best_only=False)
  ]
)

scores = classifier.evaluate_generator(test_set, steps=5)
print('%s: %.2f%%' % (
    classifier.metrics_names[1], 
    scores[1] * 100
))
