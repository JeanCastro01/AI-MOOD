import datetime
import os

import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


MX_PREC = True
ACCEL = False

LIST_GPU = tf.config.experimental.list_physical_devices('GPU')
if LIST_GPU:
    try:
        for GPU in LIST_GPU:
            tf.config.experimental.set_memory_growth(GPU, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(LIST_GPU), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as  RE:
        print(RE)

if MX_PREC:
    mx_policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(mx_policy)
    print('Mixed precision enabled')

if ACCEL:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')

dist_strateg = tf.distribute.get_strategy()
COPIES = dist_strateg.num_replicas_in_sync
print(f'REPLICAS: {COPIES}')
print("Tensorflow version " + tf.__version__)

train_dir = './train/'
test_dir = './test/'

row, col = 48, 48
classes = 7


def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))
    df = pd.DataFrame(dict_, index=[set_])
    return df


train_count = count_exp(train_dir, 'train')
test_count = count_exp(test_dir, 'test')
print(train_count)
print(test_count)

image_size = 48
batch_size = 64
data_trained = ImageDataGenerator(rescale=1. / 255,
                                  zoom_range=0.3,
                                  horizontal_flip=True)

set_training = data_trained.flow_from_directory(train_dir,
                                                batch_size=64,
                                                target_size=(image_size, image_size),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

data_tested = ImageDataGenerator(rescale=1. / 255)
set_testing = data_tested.flow_from_directory(test_dir,
                                              batch_size=64,
                                              target_size=(image_size, image_size),
                                              shuffle=True,
                                              color_mode='grayscale',
                                              class_mode='categorical')


def fst_model(input_size, classes=7):
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6),  # Compile
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def scd_model(input_size, classes=7):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Faltten the model
    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(classes, activation='softmax'))
    opt = Adam(lr=0.0001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary
    return model


model_1 = fst_model((row, col, 1), classes)

chk_path = 'model_1.h5'
log_dir = "checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint = ModelCheckpoint(filepath=chk_path,
                             save_best_only=True,
                             verbose=1,
                             mode='min',
                             moniter='val_loss')

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=3,
                           verbose=1,
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=6,
                              verbose=1,
                              min_delta=0.0001)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = CSVLogger('training.log')

callbacks = [checkpoint, reduce_lr, csv_logger]

steps_per_epoch = set_training.n // set_training.batch_size
validation_steps = set_testing.n // set_testing.batch_size

training_model = model_1.fit(x=set_training,
                             validation_data=set_testing,
                             epochs=60,
                             callbacks=callbacks,
                             steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps)
