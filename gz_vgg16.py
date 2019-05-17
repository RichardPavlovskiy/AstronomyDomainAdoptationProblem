import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import pandas as pd
import numpy as np
import time



batch_size = 128
num_classes = 2
epochs = 50
data_augmentation = False
num_predictions = 20
shift = 0.2
data_split = 0.13 #ratio of test/train
validation_split = 0.3# splits the train into train and va
model_path = './gz_vgg_with_data_aug_categorical-{}.h5'.format(int(time.time()))

callbacks = [
	ModelCheckpoint(model_path,
				monitor='val_acc',
				save_best_only=True,
				mode='max',
				verbose=0)
	]


# The data, split between test, train and val sets, a new column with smooth/disk value added for binary classification:
y = pd.read_csv('y_training_ds1.csv')
y = y.drop("Unnamed: 0", axis=1)
#y['class'] = pd.get_dummies(y['Class1.1']).values[:, ::-1].tolist() #CATEGORICAL
y['class'] = np.where(y['Class1.1']==1, "smooth", "disk")
y['filename'] = y['GalaxyID'].astype(str) + '.jpg'
y = y.drop("GalaxyID", axis=1)


a = round(data_split * int(y.shape[0]))
y_test, y_train = y.iloc[0:a,:], y.iloc[a:,:]


#batchc generators:
train_datagen = ImageDataGenerator(validation_split=0.3)
#rescale=1./255

train_generator = train_datagen.flow_from_dataframe(dataframe = y_train, directory = 'images/x_training_ds1', x_col = 'filename', y_col = 'class', class_mode = "binary", classes = ['smooth', 'disk'], batch_size = batch_size, target_size=(224,224), subset='training')
#classes = ['smooth', 'disk']
val_generator = train_datagen.flow_from_dataframe(dataframe = y_train, directory = 'images/x_training_ds1', x_col = 'filename', y_col = 'class', class_mode = "binary", classes = ['smooth', 'disk'], batch_size = batch_size, target_size=(224,224), subset='validation')

test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(dataframe = y_test, directory = 'images/x_training_ds1', x_col = 'filename', y_col = 'class', class_mode = "binary",classes = ['smooth', 'disk'], batch_size = batch_size, target_size=(224,224))



vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False


model = Sequential()

model.add(vgg_conv)

# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

opt = keras.optimizers.Adam(lr=3e-4)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


print(val_generator.samples)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.samples // batch_size),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=(val_generator.samples // batch_size),
        callbacks=callbacks)








"""
Here is the code that let's you explicitly set the ratio of test train val sets:


a = round(data_split * int(y.shape[0]))#index of last element of test set
b = a + round((1-validation_split) * (y.shape[0] - a))#index of last element of test set
print("#of test samples: " + str(a))
print("#of validation samples:" + str(0.3*(int(y.shape[0]) - a)))
y_test, y_train, y_val = y.iloc[0:a,:], y.iloc[a:b,:], y.iloc[b:,:]




#batchc generators:
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe = y_train, directory = 'images/x_training_ds1', x_col = 'filename', y_col = 'class', class_mode = "binary", batch_size = batch_size, target_size=(224,224), classes = ['smooth', 'disk'])

val_generator = train_datagen.flow_from_dataframe(dataframe = y_val, directory = 'images/x_training_ds1', x_col = 'filename', y_col = 'class', class_mode = "binary", batch_size = batch_size, target_size=(224,224), classes = ['smooth', 'disk'])

test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(dataframe = y_test, directory = 'images/x_training_ds1', x_col = 'filename', y_col = 'class', class_mode = "binary", batch_size = batch_size, target_size=(224,224), classes = ['smooth', 'disk'])
"""
