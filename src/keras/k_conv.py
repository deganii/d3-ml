# load the complex array of the reconstruction:
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
import scipy
import h5py


from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import numpy as np

BATCH_SIZE = 32
M = N = 200

# Rivenson et al, https://arxiv.org/ftp/arxiv/papers/1705/1705.04286.pdf
# Custom upsampling layer (Suppl Fig. 1 & 2)
class UpsampleLayer2D(Layer):

    def __init__(self, **kwargs):
        super(UpsampleLayer2D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_shape = self.input_spec[0].shape

        #self.input_shape = input_shape
        super(UpsampleLayer2D, self).build(input_shape)  # Be sure to call this somewhere!

    def stack_test(self):
        x = np.stack([[[11, 12], [13, 14]], [[21, 22], [23, 24]],
                      [[31, 32], [33, 34]], [[41, 42], [43, 44]]])
        # stack the first part:
        x_s1 = np.stack(x[0:2], -1)
        x_s1 = np.reshape(x_s1, [2,4])
        # stack the second part:
        x_s2 = np.stack(x[2:4], -1)
        x_s2 = np.reshape(x_s2, [2,4])

        # combine each together
        x_conc = np.concatenate((x_s1, x_s2), -2)
        x_final = np.reshape(x_conc, [4,4])

    # x (input shape) == 256/L x256/L x 64
    # output_shape = 2*256/L x2 * 256/L x 16
    def call(self, x):
        input_shape = self.input_spec[0].shape
        (batch_num, input_rows, input_cols, input_ch) = input_shape
        output_shape = self.compute_output_shape(input_shape)
        batch_output = []
        for batch in range(BATCH_SIZE):
            upsampled = []  # K.zeros(output_shape)
            for ch in range(0, input_ch, 4):
                # first two channels make the odd rows of the stack
                x_s1 = K.stack([x[batch, :, :, ch], x[batch, :, :, ch+1]], -2)
                x_s1 = K.reshape(x_s1, [input_rows, 2*input_cols])
                # last two channels make the even rows of the stack
                x_s2 = K.stack([x[:, :, :,  ch+2], x[:,:,:,ch+3]], -2)
                x_s2 = K.reshape(x_s2, [input_rows, 2*input_cols])

                # combine each together (interspersed)
                x_conc = K.concatenate((x_s1, x_s2), 0)
                x_final = K.reshape(x_conc, [1, 2*input_rows, 2*input_cols, 1])
                upsampled.append(x_final)

            # concatenate 16 upsampled channels into one tensor
            batch_img = K.concatenate(upsampled, -1)
            batch_output.append(batch_img)

        # concatenate 32 upsampled images into one output
        batch_conc = K.concatenate(batch_output, 0)
        return K.reshape(batch_conc, output_shape)

    def compute_output_shape(self, input_shape):
        return (BATCH_SIZE, int(input_shape[1] * 2), int(input_shape[2] * 2), int(input_shape[3] / 4))


input_shape = Input(shape=(M, N, 1))
#output_shape = Output(shape=(M, N, 2))

# downsample in parallel
d0 = input_shape

# TODO: try AveragePooling2D
d2 = MaxPooling2D((2,2), strides=(2,2), padding='valid')(input_shape)
d4 = MaxPooling2D((4,4), strides=(4,4), padding='valid')(input_shape)
d8 = MaxPooling2D((8,8), strides=(8,8), padding='valid')(input_shape)

c0 = Conv2D(16, (3, 3), activation='relu', padding='same')(d0)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(d2)
c4 = Conv2D(16, (3, 3), activation='relu', padding='same')(d4)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(d8)

def residualBlock(input_layer):
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    return Conv2D(16, (3, 3), activation='relu', padding='same')(conv1) + input_layer

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
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    # uconv1 = UpSampling2D(size=(2, 2))(conv1)
    # fix the batch size in order to process the custom layer
    return UpsampleLayer2D(batch_size=BATCH_SIZE)(conv1)

def upsampleChain(input_layer, num):
    if num == 0:
        return input_layer
    return upsampleChain(upsampleBlock(input_layer), num-1)

u2 = upsampleChain(r2, 1)
u4 = upsampleChain(r4, 2)
u8 = upsampleChain(r8, 3)

p0 = Conv2D(16, (3, 3), activation='relu', padding='same')(u0)
p2 = Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
p4 = Conv2D(16, (3, 3), activation='relu', padding='same')(u4)
p8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)

merged = keras.layers.concatenate([p0, p2, p4, p8], axis=1)
#merged = Flatten()(merged)

out = Conv2D(2, (3, 3), activation='relu', padding='same')(merged)


model = Model(input_shape, out)
#  plot_model(model, to_file=img_path)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# load npz
data_folder = '../../data/ds7-lymphoma/'
train_npz = np.load(data_folder + 'training.npz')
test_npz = np.load(data_folder + 'test.npz')
train_data, train_labels = train_npz['data'], train_npz['labels']
test_data, test_labels = test_npz['data'], test_npz['labels']

model.fit(train_data, train_labels,
          batch_size=BATCH_SIZE,
          epochs=100,
          verbose=1,
          validation_data=(test_data, test_labels))
score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
