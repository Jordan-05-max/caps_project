"""CNN model"""
import os

import numpy as np

from train_algo import *
from setups import *


# RANDOM_SEED = 123
RANDOM_SEED = 42

IMG_SIZE = (224, 224)

# set the parameters we want to change randomly
demo_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, rescale=1. / 255, shear_range=0.05,
                                  brightness_range=[0.1, 1.5], horizontal_flip=True, vertical_flip=True)

# use predefined function to load the image data into workspace

os.mkdir('preview')
x = X_train_crop[0]
x = x.reshape((1,) + x.shape)

i = 0
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug_img', save_format='jpg'):
	i += 1
	if i > 20:
		break

plt.imshow(X_train_crop[0])
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.show()

plt.figure(figsize=(15, 6))
i = 1

for img in os.listdir('preview/'):
	img = cv2.imread('preview/' + img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.subplot(3, 7, i)
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	i += 1
	if i > 3 * 7:
		break
plt.suptitle('Augmented Images')
plt.show()

# Removing preview folder
time.sleep(5)
shutil.rmtree('./preview', ignore_errors=True)

TRAIN_DIR = img_path + 'TRAIN_CROP/'
VAL_DIR = img_path + 'VAL_CROP/'

train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, brightness_range=[0.5, 1.5],
                                   horizontal_flip=True, vertical_flip=True, preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, color_mode='rgb', target_size=IMG_SIZE, batch_size=32, class_mode='binary',
                                                    seed=RANDOM_SEED)

validation_generator = test_datagen.flow_from_directory(VAL_DIR, color_mode='rgb', target_size=IMG_SIZE, batch_size=16, class_mode='binary',
                                                        seed=RANDOM_SEED)

"""model building"""
# load base model
# vgg19_weight_path = './keras-pretrained-models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG19(include_top=False, weights='imagenet', input_shape=IMG_SIZE+(3,))

"""+918219657384"""

Num_Classes = 1

model = Sequential()
model.add(base_model)

model.add(Conv2D(filters=16, kernel_size=9, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=16, kernel_size=9, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.45))

model.add(layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


"""for layer in base_model.layers:
	model.layers[0].trainable = False
# model.save('Trained_model_vgg19.h5')
# model.save('model_vgg19.keras')"""

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4), metrics=['binary_accuracy', 'AUC'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
model.summary()
