import os
import cv2
import imutils
import numpy as np
from tqdm import tqdm
from plotly import tools
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import iplot
from keras.applications.vgg19 import VGG19, preprocess_input

img_path = 'D:/PycharmProject/tumor_detection/brain tumor/brain_tumor_dataset/'


def load_data(dir_path, img_size=(100, 100)):
	"""
    Load resized images as np. arrays to workspace
    """
	X = []
	y = []
	i = 0
	labels = dict()
	# print(f'path:{dir_path}')
	for path in tqdm(sorted(os.listdir(dir_path))):
		if not path.startswith('.'):
			labels[i] = path
			for file in os.listdir(dir_path + path):
				if not file.startswith('.'):
					img = cv2.imread(dir_path + path + '/' + file)
					X.append(img)
					y.append(i)
			
			i += 1
	
	# X = np.array(X, dtype=np.ndarray)
	# y = np.array(y, dtype=np.ndarray)
	X = np.array(X, dtype='object')
	y = np.array(y, dtype='uint8')
	print(f'{len(X)} images loaded from {dir_path} directory.')
	# print(f'{len(y)} images loaded from {dir_path} directory.')
	# print(f'X np.array= {X} ')
	# print(f'y np.array= {y} ')
	return X, y, labels


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
     This function prints and plots the confusion matrix.
     Normalization can be applied by setting `normalize=True`.
    """
	plt.figure(figsize=(6, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	
	thresh = cm.max() / 2.
	cm = np.round(cm, 2)
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
		         horizontalalignment="center",
		         color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


def plot_samples(X, y, labels_dict, n=50):
	"""Creates a grid plot for desired number of images (n) from the specified set"""
	
	for index in range(len(labels_dict)):
		imgs = X[np.argwhere(y == index)][:n]
		j = 10
		i = int(n / j)
		
		plt.figure(figsize=(15, 6))
		c = 1
		for img in imgs:
			plt.subplot(i, j, c)
			plt.imshow(img[0])
			
			plt.xticks([])
			plt.yticks([])
			c += 1
		plt.suptitle('Tumor: {}'.format(labels_dict[index]))
		plt.show()


def crop_imgs(set_name, add_pixels_value=0):
	"""
    Finds the extreme points on the image and crops the rectangular out of them
    """
	set_new = []
	for img in set_name:
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		
		# threshold the image, then perform a series of erosions +
		# dilations to remove any small regions of noise
		thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		
		# find contours in thresholded image, then grab the largest one
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		
		# find the extreme points
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		
		ADD_PIXELS = add_pixels_value
		new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
		set_new.append(new_img)
	
	# return np.array(set_new)
	return np.array(set_new, dtype="object")


def save_new_images(x_set, y_set, folder_name):
	i = 0
	for (img, imclass) in zip(x_set, y_set):
		if imclass == 0:
			cv2.imwrite(folder_name + 'NO/' + str(i) + '.jpg', img)
		else:
			cv2.imwrite(folder_name + 'YES/' + str(i) + '.jpg', img)
		i += 1


def preprocess_imgs(set_name, img_size):
	"""
    Resize and apply VGG-19 preprocessing
    """
	set_new = []
	for img in set_name:
		img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
		set_new.append(preprocess_input(img))
	# return np.array(set_new, dtype="object")
	return np.array(set_new)


# Data import and preprocessing

TRAIN_DIR = img_path + '.b/TRAIN/'
TEST_DIR = img_path + '.b/TEST/'
VAL_DIR = img_path + '.b/VAL/'
IMG_SIZE = (224, 224)

# use predefined function to load the image data into workspace

X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

"""
Plotting the number of samples in Training, Validation and Test sets
"""

# distribution of classes among sets
y = dict()
y[0] = []
y[1] = []
for set_name in (y_train, y_val, y_test):
	y[0].append(np.sum(set_name == 0))
	y[1].append(np.sum(set_name == 1))

trace0 = go.Bar(x=['Train Set', 'Validation Set', 'Test Set'], y=y[0], name='No', marker=dict(color='#33cc33'), opacity=1)
trace1 = go.Bar(x=['Train Set', 'Validation Set', 'Test Set'], y=y[1], name='Yes', marker=dict(color='#ff3300'), opacity=1)
data = [trace0, trace1]
layout = go.Layout(title='Count of classes in each set', xaxis={'title': 'Set'}, yaxis={'title': 'Count'})
fig = go.Figure(data, layout)
# iplot(figure_or_data=fig)
fig.show()

plot_samples (X_train, y_train, labels, 30)

RATIO_LIST = []
for set in (X_train, X_test, X_val):
	for img in set:
		RATIO_LIST.append(img.shape[1] / img.shape[0])

plt.hist(RATIO_LIST)
plt.title('Distribution of Image Ratios')
plt.xlabel('Ratio Value')
plt.ylabel('Count')
plt.show()

img = cv2.imread('D:/PycharmProject/tumor_detection/brain tumor/brain_tumor_dataset/yes/Y108.jpg')
img = cv2.resize(src=img, dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# add contour on the image
img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

# add extreme points
img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

# crop
ADD_PIXELS = 0
new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

"""Let's visualize how the cropping works"""

plt.figure(figsize=(15, 6))
plt.subplot(141)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title('Step 1. Get the original image')
plt.subplot(142)
plt.imshow(img_cnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 2. Find the biggest contour')
plt.subplot(143)
plt.imshow(img_pnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 3. Find the extreme points')
plt.subplot(144)
plt.imshow(new_img)
plt.xticks([])
plt.yticks([])
plt.title('Step 4. Crop the image')
plt.show()

# apply this for each set
X_train_crop = crop_imgs(set_name=X_train)
X_val_crop = crop_imgs(set_name=X_val)
X_test_crop = crop_imgs(set_name=X_test)
plot_samples(X_train_crop, y_train, labels, 30)

save_new_images(X_train_crop, y_train, folder_name=img_path + 'TRAIN_CROP/')
save_new_images(X_val_crop, y_val, folder_name=img_path + 'VAL_CROP/')
save_new_images(X_test_crop, y_test, folder_name=img_path + 'TEST_CROP/')

X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)
plot_samples(X_train_prep, y_train, labels, 30)
# print(type(y_train))
