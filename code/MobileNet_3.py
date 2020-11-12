import re
import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import keras
from keras.preprocessing import image
from tensorflow.keras import regularizers
from keras.utils import to_categorical
from keras.applications import *
from tensorflow.keras.applications import *
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D,GlobalAveragePooling1D,MaxPooling2D,GlobalAveragePooling2D

# Transfer learning with pre-trained weights
# MobileNet with a 512-neuron layer
def MobileNet_512(img_size) :
    # Load weights pre-trained on ImageNet.
    base_model=MobileNet(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    base_model.trainable = True
    x=base_model.output
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.3)(x)
    x= BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    preds=Dense(3,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=preds)
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model

# Transfer learning with pre-trained weights
# MobileNet with a 512-neuron layer and a 256-neuron layer
def MobileNet_512_256(img_size) :
    # Load weights pre-trained on ImageNet.
    base_model=MobileNet(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    base_model.trainable = True
    x=base_model.output
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.3)(x)
    x= BatchNormalization()(x)
    x=Dense(256,activation='relu')(x)
    x=Dropout(0.3)(x)
    x= BatchNormalization()(x)
    x=GlobalAveragePooling2D()(x)
    preds=Dense(3,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=preds)
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model

# Transfer learning with pre-trained weights
# InceptionResNetV2 with a 512-neuron layer
def InceptionResNetV2_512(img_size) :
    # Load weights pre-trained on ImageNet.
    base_model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    base_model.trainable = True
    x=base_model.output
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dropout(0.3)(x)
    x= BatchNormalization()(x)
    x=GlobalAveragePooling2D()(x)
    preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model

# Transfer learning with pre-trained weights
# InceptionResNetV2 with a 512-neuron layer and a 256-neuron layer
def InceptionResNetV2_512_256(img_size) :
    # Load weights pre-trained on ImageNet.
    base_model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    base_model.trainable = True
    x=base_model.output
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dropout(0.3)(x)
    x= BatchNormalization()(x)
    x=Dense(256,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dropout(0.3)(x)
    x= BatchNormalization()(x)
    x=GlobalAveragePooling2D()(x)
    preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model

# Transfer learning with pre-trained weights
# Xception with a 512-neuron layer and a 256-neuron layer
def Xception_512_256(img_size):
    # Load weights pre-trained on ImageNet.
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    base_model.trainable = True
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    # Scale inputs to [-1, +1]
    x = norm_layer(inputs)
    norm_layer.set_weights([mean, var])

    x = base_model(x, training=False)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(0.5)(x)
    preds = Dense(3, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.8)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    # Get the current directory
    path = os.getcwd()

    # Take in the paths
    argparser = ArgumentParser()
    argparser.add_argument('train_data', help='Path to the training data')
    argparser.add_argument('test_data', help='Path to the testing data')
    args = argparser.parse_args()
    train_data_dir = Path(args.train_data)
    test_data_dir = Path(args.test_data)

    # Define the image size 512*512
    img_size = 512

    # Load the training dataset
    train_files = os.listdir(os.path.join(train_data_dir, 'train_images'))
    train_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    train_image = []
    for img_filename in train_files:
        img = image.load_img(os.path.join(train_data_dir, 'train_images', img_filename),
                             target_size=(img_size, img_size, 1))
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)
    X_train = np.array(train_image)

    trainlabel = pd.read_csv(os.path.join(train_data_dir, 'train_label.csv'))
    y_train = trainlabel['Label'].values
    y_train = to_categorical(y_train)

    # Load the testing dataset
    test_files = os.listdir(os.path.join(test_data_dir, 'test_images'))
    test_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    test_image = []
    for img_filename in test_files:
        img = image.load_img(os.path.join(test_data_dir, 'test_images', img_filename),
                             target_size=(img_size, img_size, 1))
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)
    X_test = np.array(test_image)

    # 80/20 split for training and validation dataset
    Xtrain, Xval, ytrain, yval = train_test_split(X_train, y_train, random_state=42, test_size=0.2)

    # Data Preprocessing
    train_generator = ImageDataGenerator(rotation_range=45,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         zoom_range=0.1,
                                         shear_range=0.1)

    # Data Augmentation
    train_generator.fit(Xtrain)


    # Train different models with 80/20 and predict
    for model_name in [ 'MobileNet_512_batch_4']:
        ckp_path = os.path.join(path, 'ckp', model_name+'_weights.hdf5')
        csv_path = os.path.join(path, 'ckp', 'csv', model_name + '_prediction_df.csv')

        if model_name == 'MobileNet_512_batch_4':
            img_size = 512
            reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                                          verbose=0, mode="auto", epsilon=1e-04, cooldown=0,
                                          min_lr=0.0001)
            es = EarlyStopping(monitor="val_loss", verbose=1, mode='min', patience=50)
            batch_size = 4
            model = MobileNet_512(img_size)
            model.fit(train_generator.flow(Xtrain, ytrain, batch_size=batch_size),
                                steps_per_epoch=Xtrain.shape[0] // batch_size,
                                epochs=150,
                                #epochs = 1,
                                verbose=True,
                                validation_data=(Xval, yval),
                                callbacks=[reduce_lr, es,
                                           ModelCheckpoint(filepath=ckp_path, monitor='val_accuracy', verbose=1,
                                                           save_best_only=True, mode='max')]
                                )

            # Load the weight and predict
            model = MobileNet_512(img_size)
            model.load_weights(ckp_path)
            prediction = model.predict(X_test)
            prediction_df = pd.DataFrame(prediction)
            prediction_df['Label'] = prediction_df.idxmax(axis=1)
            prediction_df = prediction_df[['Label']]
            prediction_df = prediction_df.reset_index()
            prediction_df.columns = ['ID', 'Label']
            prediction_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()