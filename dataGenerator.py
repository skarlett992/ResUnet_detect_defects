import cv2
from skimage import io
from sklearn.model_selection import train_test_split
import os
# %matplotlib inline
from utilities import rle2mask
from keras_preprocessing.image import ImageDataGenerator


def dataGenerator(range_i, train_dir, defect_class_mask_df,all_images_df):
    for i in range(range_i):

      # Read the images using opencv and converting to rgb format
      img = io.imread(os.path.join(train_dir, defect_class_mask_df.ImageId[i]))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Get the mask for the image from rle
      mask = rle2mask(defect_class_mask_df.EncodedPixels[i], img.shape[0], img.shape[1])

      train, test = train_test_split(all_images_df, test_size=0.15)

      # Create a data generator which scales the data from 0 to 1 and makes validation split of 0.15
      datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.15)

      train_generator = datagen.flow_from_dataframe(
          dataframe=train,
          directory=train_dir,
          x_col="ImageID",
          y_col="label",
          subset="training",
          batch_size=16,
          shuffle=True,
          class_mode="other",
          target_size=(256, 256))  # Input size required by ResNet model

      valid_generator = datagen.flow_from_dataframe(
          dataframe=train,
          directory=train_dir,
          x_col="ImageID",
          y_col="label",
          subset="validation",
          batch_size=16,
          shuffle=True,
          class_mode="other",
          target_size=(256, 256), )
      test_datagen = ImageDataGenerator(rescale=1. / 255.)

      test_generator = test_datagen.flow_from_dataframe(
          dataframe=test,
          directory=train_dir,
          x_col="ImageID",
          y_col=None,
          batch_size=16,
          shuffle=False,
          class_mode=None,
          target_size=(256, 256))
    return train_generator, valid_generator, test_generator, train, test, mask