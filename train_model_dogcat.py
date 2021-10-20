import numpy as np
from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

img_width,img_height = 150,150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000  # ==len(data/train)
nb_validation_samples = 800
epochs = 30
batch_size = 16
if K.image_data_format()=='channels_first' :
    input_shape =(3,img_width,img_height)
else :
    input_shape = (img_width,img_height,3)

#build model
model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
#model.add(Activation('sigmoid'))
model.summary()

#train model
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.1,zoom_range=0.1,horizontal_flip=True) #them data train
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size = batch_size,
    class_mode = 'binary')
print(train_generator.class_indices)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = 'binary')
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
model.save('model_dog_cat.h5')