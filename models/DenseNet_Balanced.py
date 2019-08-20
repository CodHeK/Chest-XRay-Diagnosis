import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.models import Sequential

from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


dtrain=pd.read_csv("CheXpert-v1.0-small/train.csv")
dtrain = dtrain.fillna(0)

dnew=pd.read_csv("CheXpert-v1.0-small/valid.csv")
dnew = dnew.fillna(0)

#add dnew to dtrain to re-split since valid data in data set is very small
dtrain = dtrain.append(dnew)

#pre-process data: remove Lateral images we deal with only frontal images
dtrain = dtrain[~dtrain[dtrain.columns[3]].str.contains("Lateral")]
#pre-process data: drop selected features - only images as inputs
dtrain = dtrain.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

print(dtrain.shape)
dtrain.describe().transpose()

# dealing with uncertanty (-1) values
dtrain = dtrain.replace(-1,1)
dtrain.describe().transpose()

dtrain_upsample=[]
dtrain_upsample_list=[]

dtrain_upsample_list = [features_data[0],features_data[0],features_data[0],features_data[0],
                       features_data[1],
                       features_data[2],
                       features_data[4],features_data[4],features_data[4],features_data[4],features_data[4],features_data[4],features_data[4],features_data[4],features_data[4],
                       features_data[5],
                       features_data[6],
                       features_data[7],
                       features_data[8],
                       features_data[9],
                       features_data[11],features_data[11],features_data[11],features_data[11],features_data[11],features_data[11],features_data[11],features_data[11],features_data[11],features_data[11],
                       features_data[12],features_data[12],features_data[12],features_data[12],features_data[12],features_data[12],features_data[12],features_data[12]]

dtrain_upsample = pd.concat(dtrain_upsample_list)
print(dtrain_upsample.shape)
print(list(dtrain_upsample.columns[1:15]))

features_sizeR=[]
features_dataR =[]
features_nameR=[]
#print(list(dtrain.columns[1:15]))
for featureR in list(dtrain_upsample.columns[1:15]):
    data_featureR = dtrain_upsample.loc[dtrain_upsample[featureR] == 1]
    features_sizeR.append(data_featureR.shape[0])
    features_dataR.append(data_featureR)
    features_nameR.append(featureR)

objectsR = list(dtrain_upsample.columns[1:15])
y_posR = np.arange(len(objectsR))
performanceR = np.array(features_sizeR)/dtrain_upsample.shape[0]*100

# split data into train/valid/test
# shuffle data
dtrain_upsample = dtrain_upsample.sample(frac=1)
# split data
dvalid_size = round(0.1*dtrain_upsample.shape[0])
dtest_size = dvalid_size
dtr = dtrain_upsample[0:dtrain_upsample.shape[0]-dvalid_size-dtest_size+1]
dv = dtrain_upsample[dtrain_upsample.shape[0]-dvalid_size-dtest_size:dtrain_upsample.shape[0]-dvalid_size+1]
dte = dtrain_upsample[dtrain_upsample.shape[0]-dvalid_size:dtrain_upsample.shape[0]+1]

print(dtr.shape)
print(dv.shape)
print(dte.shape)

# data generation for Keras
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255.)
valid_datagen=ImageDataGenerator(rescale=1./255.)

target_size = (224,224)
train_generator=train_datagen.flow_from_dataframe(dataframe=dtr, directory=None , x_col="Path", y_col=list(dtr.columns[1:15]), class_mode="other", drop_duplicates = False, target_size=target_size, batch_size=32)
valid_generator=valid_datagen.flow_from_dataframe(dataframe=dv, directory=None, x_col="Path", y_col=list(dv.columns[1:15]), class_mode="other", drop_duplicates = False, target_size=target_size, batch_size=32)
test_generator=test_datagen.flow_from_dataframe(dataframe=dte, directory=None, x_col="Path", y_col=list(dte.columns[1:15]), class_mode="other", drop_duplicates = False, target_size=target_size, shuffle = False, batch_size=1)

### model architecture design/selection
# create the base pre-trained model
base_model = DenseNet121(include_top = False, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(14, activation='sigmoid')(x)


# this is the model we will train
model_F = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
   layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

#model training

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_F.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print(model_F.summary())

### fit model
num_epochs = 3
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model_H = model_F.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=num_epochs)
# save model
model_F.save("model_DenseNet_Balanced.h5")

model_F = load_model('model_DenseNet_Balanced.h5')
num_epochs = 3
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# prediction and performance assessment
test_generator.reset()
pred=model_F.predict_generator(test_generator, steps=STEP_SIZE_TEST)
pred_bool = (pred >= 0.5)

y_pred = np.array(pred_bool,dtype =int)

dtest = dte.to_numpy()
y_true = np.array(dtest[:,1:15],dtype=int)

print(classification_report(y_true, y_pred,target_names=list(dtr.columns[1:15])))

score, acc = model_F.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
print('Test score:', score)
print('Test accuracy:', acc)
