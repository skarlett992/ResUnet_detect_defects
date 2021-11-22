import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def resblock(X, f):
  X_copy = X
  X = Conv2D(f, kernel_size = (1,1), strides = (1,1), kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)

  X = Conv2D(f, kernel_size = (3,3), strides =(1,1), padding = 'same', kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X_copy = Conv2D(f, kernel_size = (1,1), strides =(1,1), kernel_initializer ='he_normal')(X_copy)
  X_copy = BatchNormalization()(X_copy)
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X

def upsample_concat(x, skip):
  x = UpSampling2D((2,2))(x)
  merge = Concatenate()([x, skip])
  return merge

def last_layers(X_input):
  # Stage 1
  conv1_in = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)
  conv1_in = BatchNormalization()(conv1_in)
  conv1_in = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_in)
  conv1_in = BatchNormalization()(conv1_in)
  pool_1 = MaxPool2D(pool_size=(2, 2))(conv1_in)

  # Stage 2
  conv2_in = resblock(pool_1, 32)
  pool_2 = MaxPool2D(pool_size=(2, 2))(conv2_in)

  # Stage 3
  conv3_in = resblock(pool_2, 64)
  pool_3 = MaxPool2D(pool_size=(2, 2))(conv3_in)

  # Stage 4
  conv4_in = resblock(pool_3, 128)
  pool_4 = MaxPool2D(pool_size=(2, 2))(conv4_in)

  # Stage 5
  conv5_in = resblock(pool_4, 256)

  # Upscale stage 1
  up_1 = upsample_concat(conv5_in, conv4_in)
  up_1 = resblock(up_1, 128)

  # Upscale stage 2
  up_2 = upsample_concat(up_1, conv3_in)
  up_2 = resblock(up_2, 64)

  # Upscale stage 3
  up_3 = upsample_concat(up_2, conv2_in)
  up_3 = resblock(up_3, 32)

  # Upscale stage 4
  up_4 = upsample_concat(up_3, conv1_in)
  up_4 = resblock(up_4, 16)

  # Final Output
  output = Conv2D(4, (1, 1), padding="same", activation="sigmoid")(up_4)

  model_seg = Model(inputs=X_input, outputs=output)

  adam = tf.keras.optimizers.Adam(lr=0.05, epsilon=0.1)
  return adam, model_seg

def basemodel_last_layers(basemodel):
  for layer in basemodel.layers:
    layer.trainable = False

  headmodel = basemodel.output
  headmodel = AveragePooling2D(pool_size=(4, 4))(headmodel)
  headmodel = Flatten(name='flatten')(headmodel)
  headmodel = Dense(256, activation="relu")(headmodel)
  headmodel = Dropout(0.3)(headmodel)
  headmodel = Dense(1, activation='sigmoid')(headmodel)

  model = Model(inputs=basemodel.input, outputs=headmodel)
  return model