import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import np_utils
# from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.optimizers import Adam

dataset_images = pd.read_csv('fer2013.csv')


X_train, train_y, X_test, test_y = [], [], [], []

for l, row in dataset_images.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{l} and row:{row}")

total_feat = 64
total_lab = 7
batch_size = 64
epochs = 60
width, height = 48, 48

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

train_y = np_utils.to_categorical(train_y, num_classes=total_lab)
test_y = np_utils.to_categorical(test_y, num_classes=total_lab)

# normalize data
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

##designing the CNN

trained_model = Sequential()

# 1st layer
trained_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
trained_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
trained_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
trained_model.add(Dropout(0.5))

# 2nd layer
trained_model.add(Conv2D(64, (3, 3), activation='relu'))
trained_model.add(Conv2D(64, (3, 3), activation='relu'))
trained_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
trained_model.add(Dropout(0.5))

# 3rd layer
trained_model.add(Conv2D(128, (3, 3), activation='relu'))
trained_model.add(Conv2D(128, (3, 3), activation='relu'))
trained_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

trained_model.add(Flatten())

# CNN
trained_model.add(Dense(1024, activation='relu'))
trained_model.add(Dropout(0.2))
trained_model.add(Dense(1024, activation='relu'))
trained_model.add(Dropout(0.2))

trained_model.add(Dense(total_lab, activation='softmax'))

# Complile the model (ready)
trained_model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Train the model
trained_model.fit(X_train, train_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test, test_y),
                  shuffle=True)

# Save the model
fer_json_file = trained_model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json_file)
trained_model.save_weights("fer.h5")
