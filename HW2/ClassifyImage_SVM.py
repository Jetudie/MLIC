import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.svm import SVC

import feature
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage


# Specify the directory and file name pattern
CURRENT_DIR = os.getcwd()
TEST_IMAGE_DIR = os.path.join(CURRENT_DIR, 'CSL', 'test')
TRAINING_IMAGE_DIR = os.path.join(CURRENT_DIR, 'CSL', 'training')

FILE_PATTERN = r'[a-z]\d{3}' + '.jpg'
TEST_FILES = re.findall(FILE_PATTERN, str(os.listdir(TEST_IMAGE_DIR)))
TRAINING_FILES = re.findall(FILE_PATTERN, str(os.listdir(TRAINING_IMAGE_DIR)))

SAVE_DIR = os.path.join(CURRENT_DIR, 'save')

a = 0
# Save label and feature of images
def save_label_feature(Dir ,Files):
    Label = []
    x = np.empty((0,26244))
    i = 0
    j = 0
    for f in Files:
        label = f[0]
        Label.append(label)
        img = plt.imread(os.path.join(Dir, f), format='jpeg')
        grayify = feature.RGB2GrayTransformer()
        hogify = feature.HogTransformer(
            pixels_per_cell=(8, 8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys'
            )
        scalify = StandardScaler()
        X_train_gray = grayify.fit_transform(img)
        X_train_hog = hogify.fit_transform(X_train_gray)
        x = np.vstack((x, scalify.fit_transform(X_train_hog)))
        print(str(f) + " finished")
        i += 1
        if i == 300:
            np.save('x_file' + str(a) + str(j), x)
            j += 1
            i = 0
            x = np.empty((0,26244))
    np.save('x_file' + str(a) + str(j), x)
    x = np.empty((0,26244))
    y = np.asarray(Label)
    for k in range(j):
        arr = np.load('x_file' + str(a) + str(k) + '.npy')
        x = np.concatenate((x, arr))
    #y.concatenate(np.asarray(Label))
    np.save('X_file' + str(a) + str(j), x)
    np.save('Y_file' + str(a) + str(j), y)
    print(x.shape)
    print(y.shape)
    print(y)
    return x, y

def classify_svm(data, label):
    clf = SVC(kernel = 'poly')
    clf.fit(data,label)
    return clf

if __name__ == "__main__":
    x_train, y_train = save_label_feature(TRAINING_IMAGE_DIR, TRAINING_FILES)
    classifier = classify_svm(x_train, y_train)
    a += 1
    x_test, y_test = save_label_feature(TEST_IMAGE_DIR, TEST_FILES)
    ans = classifier.predict(x_test)
    print(ans)



