# load the complex array of the reconstruction:
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
import scipy
import h5py

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Rivenson et al, https://arxiv.org/ftp/arxiv/papers/1705/1705.04286.pdf
# Custom upsampling layer (Suppl Fig. 1 & 2)
class UpsampleLayer2D(Layer):

    def __init__(self, **kwargs):
        super(UpsampleLayer2D, self).__init__(**kwargs)


    def build(self, input_shape):
        super(UpsampleLayer2D, self).build(input_shape)  # Be sure to call this somewhere!

    # x (input shape) == 1568/L x1365/L x 64
    # output_shape = 2*1568/L x2 * 1365/L x 16
    def call(self, x):
        output_shape = self.compute_output_shape(self.input_shape)
        upsampled = np.zeros(output_shape)

        # 1 to 64
        for z in range(input_shape[3]):
            for j in range(input_shape[2]):
                for i in range(input_shape[1]):
                    remainder = i % 4
                    offsetX = 0
                    offsetY = 0
                    if remainder == 1 or remainder == 3:
                        offsetX = offsetX + 1
                    elif remainder == 2 or remainder == 3:
                        offsetY = offsetY + 1
                    upsampled[2*i + offsetX , 2*j + offsetY, np.floor(z/4)] = x[i, j, z]
        return upsampled

    def compute_output_shape(self, input_shape):
        return (input_shape[0] * 2, input_shape[1] * 2, input_shape[3] / 4)

#
# sample_data = h5py.File('C:/dev/courses/6.874/Final-Project/src/images/Daudi_PS_NAV_Postwash_image1.mat', 'r')
# # load a sample image
# subNormAmp = sample_data['subNormAmp']
# reconImage = sample_data['ReconImage']
# M = subNormAmp.shape[0]
# N = subNormAmp.shape[1]
#reshaped = [[reconImage[m,n] for m in range(reconImage.shape[0])] for n in range(reconImage.shape[1])]

# load npz

data_folder = '../../data/ds7-lymphoma/'
train_npz = np.load(data_folder + 'training.npz')
train_data, train_labels = train_npz['data'], train_npz['labels']

random.choice(mylist,3)



input_shape = Input(shape=(M, N, 1))
#output_shape = Output(shape=(M, N, 2))

# downsample in parallel
d0 = input_shape
d2 = MaxPooling2D((2,2), strides=(2,2), padding='valid')(input_shape)
d4 = MaxPooling2D((4,4), strides=(4,4), padding='valid')(input_shape)
d8 = MaxPooling2D((8,8), strides=(8,8), padding='valid')(input_shape)

c0 = Conv2D(16, (3, 3), activation='relu')(d0)
c2 = Conv2D(16, (3, 3), activation='relu')(d2)
c4 = Conv2D(16, (3, 3), activation='relu')(d4)
c8 = Conv2D(16, (3, 3), activation='relu')(d8)

def residualBlock(input_layer):
    conv1 = Conv2D(16, (3, 3), activation='relu')(input_layer)
    return Conv2D(16, (3, 3), activation='relu')(conv1) + input_layer

def residualBlockChain(input_layer):
    r0 = input_layer
    for i in range(4):
        r0 = residualBlock(r0)
    return r0

r0 = residualBlockChain(c0)
r2 = residualBlockChain(c2)
r4 = residualBlockChain(c4)
r8 = residualBlockChain(c8)

u0 = r0

def upsampleBlock(input_layer):
    conv1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    # uconv1 = UpSampling2D(size=(2, 2))(conv1)
    return UpsampleLayer2D(conv1)

def upsampleChain(input_layer, num):
    if num == 0:
        return input_layer
    return upsampleChain(upsampleBlock(input_layer), num-1)

u2 = upsampleChain(r2, 1)
u4 = upsampleChain(r4, 2)
u8 = upsampleChain(r8, 3)

p0 = Conv2D(16, (3, 3), activation='relu')(u0)
p2 = Conv2D(16, (3, 3), activation='relu')(u2)
p4 = Conv2D(16, (3, 3), activation='relu')(u4)
p8 = Conv2D(16, (3, 3), activation='relu')(u8)

merged = keras.layers.concatenate([p0, p2, p4, p8], axis=1)
#merged = Flatten()(merged)

out = Conv2D(2, (3, 3), activation='relu')(merged)


model = Model(input_shape, out)
#  plot_model(model, to_file=img_path)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# load data


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])