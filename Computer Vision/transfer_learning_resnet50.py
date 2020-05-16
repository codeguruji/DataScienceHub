## Transfer learning using ResNet50 weights
## https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38

# ResNet50 is a 50 layers residual network
# Residual is nothing but skip connections

import glob
import numpy as np
import pandas as pd
import os
import shutil #for copying files and directories
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.regularization import l2
%matplotlib inline

#we will train the classifier to identify between cat and dog

#lets read the cat and dog data files
files = glob.glob(r'C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog\train\*')

cat_files = [fn for fn in files if 'cat.' in fn]
dog_files = [fn for fn in files if 'dog.' in fn] 
len(cat_files), len(dog_files)

#lets take a small sample of the train and test data
cat_train = np.random.choice(cat_files, size=1500, replace=False) 
dog_train = np.random.choice(dog_files, size=1500, replace=False)


cat_files = list(set(cat_files) - set(cat_train)) 
dog_files = list(set(dog_files) - set(dog_train)) 
 
cat_val = np.random.choice(cat_files, size=500, replace=False)
dog_val = np.random.choice(dog_files, size=500, replace=False)
cat_files = list(set(cat_files) - set(cat_val))
dog_files = list(set(dog_files) - set(dog_val)) 
 
cat_test = np.random.choice(cat_files, size=500, replace=False) 
dog_test = np.random.choice(dog_files, size=500, replace=False) 
 
print("Cat datasets:", cat_train.shape, cat_val.shape, cat_test.shape) 
print("Dog datasets:", dog_train.shape, dog_val.shape, dog_test.shape)


## Create directories for train and test labels
def create_class_label_dirs(paths, base_dir, data_type="train"):
    base_dir = os.path.join(base_dir, data_type)
    for p in paths:
        dir = os.path.dirname(p)
        label = p.split("\\")[-1].split(".")[0]
        if data_type=="test":
            label=""
        if not os.path.exists(os.path.join(base_dir, label)):
            os.makedirs(os.path.join(base_dir, label))
        shutil.move(p, os.path.join(base_dir, label, os.path.relpath(p, dir)))

create_class_label_dirs(cat_train, r"C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog", "train")
create_class_label_dirs(dog_train, r"C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog", "train")

create_class_label_dirs(cat_val, r'C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog', "validation")
create_class_label_dirs(dog_val, r'C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog', "validation")

create_class_label_dirs(cat_test, r'C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog', "test")
create_class_label_dirs(dog_test, r'C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\cat dog', "test")

## Data augmentation
#keras has a pretty good utility for real time data augmentation
#it scales, rotates and translates (affine transforms) the images randomly to create new datasets
train_datagen = ImageDataGenerator(rescale=1./255,\
                                   zoom_range=0.3,#scale up and down by a factor of 0.3\
                                   rotation_range=50, #rotate by 50 degree\
                                   width_shift_range=0.2, #translate horizontally\
                                   height_shift_range=0.2, #translate vertically\
                                   shear_range=0.2, #apply shear transforms\
                                   horizontal_flip=True, #random flipping\
                                   fill_mode='nearest')#fill the blank pixels created as a result of above transforms with the nearest pixels

test_datagen = ImageDataGenerator(rescale=1./255)

## test the generator
# cat_generator = train_datagen.flow(np.array([img_to_array(load_img(train_files[1221], target_size=IMG_DIM))]),
# train_labels[1221:1221+1], batch_size=1)
# cat = [next(cat_generator) for i in range(0,5)]
# fig, ax = plt.subplots(1,5, figsize=(16, 6))
# print('Labels:', [item[1][0] for item in cat]) 
# l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
# plt.show()

#lets convert all the images to 300X300 RGB images
IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_HEIGHT, IMG_WIDTH)

train_generator = train_datagen.flow_from_directory(directory="./train",\
                                                    target_size=IMG_DIM,\
                                                    batch_size=32,\
                                                    class_mode='binary')
val_generator = train_datagen.flow_from_directory(directory="./validation", target_size=IMG_DIM, batch_size=32, class_mode='binary')

test_generator = test_datagen.flow_from_directory(directory="./test", target_size=IMG_DIM, batch_size=32, class_mode=None)


## Transfer learning from pre-trained model for feature extraction
## ResNet50

#don't load the last fully connected layer
#also we freeze the weights of the model by setting trainable = False (for feature extractor)
#we don't want to change the knowledge learnt from ImageNet data

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()

## now lets add our own fully connected layer and sigmoid for the output layer (binary)
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=input_shape, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, 
                              epochs=100,
                              validation_data=val_generator, 
                              validation_steps=50, 
                              verbose=1)

model.save('cats_dogs_tlearn_img_aug_cnn_restnet50.h5')

##plot the learning accuracy on epochs to see the convergence

## Now lets fine-tune the initial few convolution layers by unfreezing the weights
restnet.trainable = True
set_trainable = False
for layer in restnet.layers:
    if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])


model_finetuned = Sequential()
model_finetuned.add(restnet)
model_finetuned.add(Dense(512, activation='relu', input_dim=input_shape))
model_finetuned.add(Dropout(0.3))
model_finetuned.add(Dense(512, activation='relu'))
model_finetuned.add(Dropout(0.3))
model_finetuned.add(Dense(1, activation='sigmoid'))
model_finetuned.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
model_finetuned.summary()

history_1 = model_finetuned.fit_generator(train_generator, 
                                  steps_per_epoch=100, 
                                  epochs=2,
                                  validation_data=val_generator, 
                                  validation_steps=100, 
                                  verbose=1)

model.save('cats_dogs_tlearn_finetune_img_aug_restnet50.h5')


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()

from keras.regularizers.
