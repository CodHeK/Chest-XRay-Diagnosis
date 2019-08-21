import numpy as np
import keras
import cv2
from keras.models import Model, load_model
import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

model = load_model('model_DenseNet_Balanced.h5')

target_size = (224,224)
dvalid_set=pd.read_csv("CheXpert-v1.0-small/valid.csv")
dvalid_set = dvalid_set.fillna(0)
dvalid_set = dvalid_set[~dvalid_set[dvalid_set.columns[3]].str.contains("Lateral")]
dvalid_set= dvalid_set.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)
dvalid_set = dvalid_set.replace(-1,1)
valid_datagen=ImageDataGenerator(rescale=1./255.)
valid_generator=valid_datagen.flow_from_dataframe(dataframe=dvalid_set, directory=None, x_col="Path", \
                                                  y_col=list(dvalid_set.columns[1:15]), \
                                                  class_mode="other", drop_duplicates = False, \
                                                  target_size=target_size, shuffle = False, batch_size=1)

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
valid_generator.reset()
pred=model.predict_generator(valid_generator, steps=STEP_SIZE_VALID)
pred_bool = (pred >= 0.5)
y_pred = np.array(pred_bool,dtype =int)

dvalid = dvalid_set.to_numpy()
y_true = np.array(dvalid[:,1:15],dtype=int)

print(y_true[index])
print(y_pred[index])

print(model.summary())

keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='dot', format='svg'))
#create your model
#then call the function on your model
visualize_model(model)

last_conv_layer = model.get_layer('relu')
print(last_conv_layer)

# The highest predicted probability observation
argmax = np.argmax(pred[index])
print(pred[index])
print(argmax)

output = model.output[:, argmax]
grads = K.gradients(output, last_conv_layer.output)[0]
print(grads)
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

x=valid_generator[index][0]
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(1024):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()

# select the sample and read the corresponding image and label
sample_image = cv2.imread(fname)
# pre-process the image
sample_image = cv2.resize(sample_image, (224,224))
if sample_image.shape[2] ==1:
    sample_image = np.dstack([sample_image, sample_image, sample_image])
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
sample_image = sample_image.astype(np.float32)/255.
sample_label = 1

sample_image_processed = np.expand_dims(sample_image, axis=0)
print(sample_image_processed.shape)

heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
heatmap = heatmap *255
heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

from skimage import data, color, io, img_as_float
sample_image_hsv = color.rgb2hsv(sample_image)
heatmap = color.rgb2hsv(heatmap)

alpha=0.7
sample_image_hsv[..., 0] = heatmap[..., 0]
sample_image_hsv[..., 1] = heatmap[..., 1] * alpha

img_masked = color.hsv2rgb(sample_image_hsv)

f,ax = plt.subplots(1,2, figsize=(16,6))
ax[0].imshow(sample_image)
ax[0].set_title(f"Image - Consolidation")
ax[0].axis('off')

ax[1].imshow(img_masked)
ax[1].set_title("Class Activation Map - Consolidation has highest predicted probability")
ax[1].axis('off')

plt.show()
