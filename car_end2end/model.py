import csv
import cv2
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D

def read_log(filepath):
  lines = []
  with open(filepath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

  return lines


# Read original data
lines_twolaps = read_log('../data2/driving_log.csv')
lines_curves = read_log('../data_curves/driving_log.csv')
lines = np.array(lines_twolaps + lines_curves)

# Balance data
nbins = 2000
max_examples = 200
balanced = np.empty([0, lines.shape[1]], dtype=lines.dtype)
for i in range(0, nbins):
  begin = i * (1.0 / nbins)
  end = begin + 1.0 / nbins
  extracted = lines[(abs(lines[:,3].astype(float)) >= begin) & (abs(lines[:,3].astype(float)) < end)]
  np.random.shuffle(extracted)
  extracted = extracted[0:max_examples, :]
  balanced = np.concatenate((balanced, extracted), axis=0)

# Prepare and augment data 
imgs, angles = [], []
offset = 0.2
correction = [0, offset, -offset]  # center, left, right cameras
for line in balanced:
  for i in range(3):
    img_path = line[i]
    img = cv2.imread(img_path)
    imgs.append(img)

    angle = float(line[3])
    angles.append(angle + correction[i])

flip_imgs, flip_angles = [], []
for img, angle in zip(imgs, angles):
  flip_imgs.append(cv2.flip(img, 1))
  flip_angles.append(-1.0 * angle)

augmented_imgs = imgs + flip_imgs
augmented_angles = angles + flip_angles

X_train = np.array(augmented_imgs)
y_train = np.array(augmented_angles)

# Build the model 
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001))
best_model = ModelCheckpoint('model_best.h5', verbose=2, save_best_only=True)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=30, callbacks=[best_model])

model.save('model_last.h5')
