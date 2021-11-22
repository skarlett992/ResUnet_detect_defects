import matplotlib.pyplot as plt
import cv2
from skimage import io
import os
# %matplotlib inline
from utilities import rle2mask , mask2rle

def drow_defect_class_mask_df(train_dir, defect_class_mask_df):
    for i in range(10):
      # Specify the path to the images given their image ID
      img = io.imread(os.path.join(train_dir, defect_class_mask_df.ImageId[i]))
      plt.figure()
      plt.title('Defect type:'+str(defect_class_mask_df.ClassId[i]))
      plt.imshow(img)


def draw_test_images(range_i, train_dir, test_df, axes, directory, df_pred):
  for i in range(range_i):
    # read the images using opencv and convert them to rgb format
    img = io.imread(os.path.join(train_dir, test_df.ImageId[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read the images using opencv and convert to rgb format
    pred_img = io.imread(os.path.join(directory, df_pred.ImageId[i]))
    pred_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Obtain mask for the image from rle
    mask = rle2mask(test_df.EncodedPixels[i], img.shape[0], img.shape[1])

    # get the mask for the image from rle
    mask_pred = rle2mask(df_pred.EncodedPixels[i], img.shape[0], img.shape[1])

    img[mask == 1, 1] = 255
    pred_img[mask_pred == 1, 0] = 255

    plt.title(test_df.ClassId[i])

    axes[2 * i].set_title(test_df.ClassId[i])
    axes[2 * i].imshow(img)
    axes[2 * i + 1].set_title(df_pred.ClassId[i])
    axes[2 * i + 1].imshow(pred_img)
    plt.tight_layout()