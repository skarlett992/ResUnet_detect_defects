import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utilities import DataGenerator
from utilities import focal_tversky, tversky
from utilities import prediction
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
# %matplotlib inline
from utilities import rle2mask
from dataGenerator import dataGenerator
from build_models.layers import last_layers, basemodel_last_layers
from plots import drow_defect_class_mask_df, draw_test_images

defect_class_mask_df = pd.read_csv('severstal-steel-defect-detection/train.csv')
all_images_df = pd.read_csv('defect_and_no_defect.csv')
defect_class_mask_df['mask'] = defect_class_mask_df['ClassId'].map(lambda x: 1)
train_dir = 'severstal-steel-defect-detection/train_images'
drow_defect_class_mask_df(train_dir, defect_class_mask_df)

image_index = 20
img = io.imread(os.path.join(train_dir, defect_class_mask_df.ImageId[image_index]))
mask = rle2mask(defect_class_mask_df.EncodedPixels[image_index], img.shape[0], img.shape[1])

batch = 10
train_generator, valid_generator, test_generator, train, test, mask = \
  dataGenerator(batch, train_dir, defect_class_mask_df, all_images_df)

basemodel = ResNet50(
    weights = 'imagenet',
    include_top = False,
    input_tensor = Input(shape=(256,256,3))
)
basemodel.summary()

model = basemodel_last_layers(basemodel)
model.compile(
    loss = 'binary_crossentropy',
    optimizer='Nadam',
    metrics= ["accuracy"]
)
earlystopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=20
)

# save the best model with least validation loss
checkpointer = ModelCheckpoint(
    filepath="build_models/resnet-weights.hdf5",
    verbose=1,
    save_best_only=True
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch= train_generator.n // 16,
    epochs = 40,
    validation_data= valid_generator,
    validation_steps= valid_generator.n // 16,
    callbacks=[checkpointer, earlystopping]
)

model_json = model.to_json()
with open("build_models/resnet-classifier-model.json", "w") as json_file:
  json_file.write(model_json)

with open('build_models/resnet-classifier-model.json', 'r') as json_file:
    json_savedModel = json_file.read()

# load the model
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('build_models/resnet-weights.hdf5')
model.compile(
    loss = 'binary_crossentropy',
    optimizer='Nadam',
    metrics= ["accuracy"]
)

test_predict = model.predict(
    test_generator,
    steps = test_generator.n // 16,
    verbose =1
)
predict = []
predict = np.asarray([predict.append(1) for i in test_predict if i >= 0.01])
original = np.asarray(test.label)[:1936]
accuracy = accuracy_score(original, predict)

cm = confusion_matrix(original, predict)
plt.figure(figsize = (7,7))
sns.heatmap(cm, annot=True)

report = classification_report(original,predict, labels = [0,1])
print(report)

X_train, X_val = train_test_split(
    defect_class_mask_df,
    test_size=0.2
)
train_ids = list(X_train.ImageId)
train_class = list(X_train.ClassId)
train_rle = list(X_train.EncodedPixels)

val_ids = list(X_val.ImageId)
val_class = list(X_val.ClassId)
val_rle = list(X_val.EncodedPixels)


training_generator = DataGenerator(train_ids,train_class, train_rle, train_dir)
validation_generator = DataGenerator(val_ids,val_class,val_rle, train_dir)

input_shape = (256,256,1)

#Input tensor shape
X_input = Input(input_shape)

adam, model_seg = last_layers(X_input)
model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])

earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="weights", verbose=1, save_best_only=True)

history = model_seg.fit_generator(training_generator,
                                  epochs = 40, validation_data= validation_generator,
                                  callbacks=[checkpointer, earlystopping])

model_json = model_seg.to_json()
with open("build_models/resunet-segmentation-model.json", "w") as json_file:
  json_file.write(model_json)

test_df = pd.read_csv('test.csv')

# make prediction
image_id, defect_type, mask = prediction(test_df, model, model_seg)

df_pred= pd.DataFrame(
    {'ImageId': image_id,
     'EncodedPixels': mask,
     'ClassId': defect_type}
)
directory = "train_images"
fig, axes = plt.subplots(10, 2, figsize = (14, 14))
axes = axes.ravel()    # convert 2D axes into 1D for below for loop
print('\tGround truth', '\t\t\t\t\tPrediction')

draw_test_images(batch, train_dir, test_df, axes, directory, df_pred)

