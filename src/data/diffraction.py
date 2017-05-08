
from __future__ import print_function
import PIL.ImageOps
import scipy
from PIL import Image, ImageDraw
import random
import numpy as np
import os
import ntpath
from scipy import misc
import glob
# generator of labeled bright-field test images
from data import BrightfieldGenerator

class DiffractionGenerator(object):

    @classmethod
    def generateImage(cls, destination, seq, bf_image, label, pad_width = 20, save = True, dx = 0.8,  dy = 0.8, z = 0.2, lmbda = 405):
        if type(bf_image) is not np.ndarray:
            U0 = np.array(bf_image)
        else:
            U0 = bf_image

        U0 = np.pad(U0, pad_width=pad_width, mode='constant', constant_values=255 )

        k = 2 * np.pi / lmbda
        ny, nx = U0.shape
        Lx = dx * nx
        Ly = dy * ny

        dfx = 1. / Lx
        dfy = 1. / Ly

        u = np.outer(np.ones(nx), [nxi - nx / 2 for nxi in range(0, nx)]) * dfx
        v = np.outer(np.ones(ny), [nyi - ny / 2 for nyi in range(0, ny)]) * dfy
        v = np.transpose(v)

        O = np.fft.fftshift(np.fft.fft2(U0))

        H = np.exp(1j * k * z) * np.exp(-1j * np.pi * lmbda * z * (np.square(u) + np.square(v)))

        U = np.fft.ifft2(np.fft.ifftshift(O * H))

        U = np.absolute(U)

        if save:
            if not os.path.exists(destination):
                os.makedirs(destination)
            scipy.misc.imsave(os.path.join(
                destination, 'D{0:05}-{1}.png'.format(seq, label )), U)
        return U, label

    @classmethod
    def generateNewImagePair(cls, destination, seq, save=True, dx=0.8, dy=0.8, z=0.2, lmbda=405):
        bf_image, label = BrightfieldGenerator.generateImage(destination,seq,save=save)
        return DiffractionGenerator.generateImage( destination, seq, bf_image, label, save=save, dx=dx, dy=dy, z=z, lmbda=lmbda)

    @classmethod
    def generateDiffractionDataset(cls, source_data, source_labels, destination, npzName, save=True, dx=0.8, dy=0.8, z=0.2, lmbda=405):
        #bf_image, label = BrightfieldGenerator.generateImage(destination,seq,save=save)
        # load source dataset
        num_images = source_data.shape[0]
        dimage_size = 80 * 80
        data = np.zeros([num_images, dimage_size], dtype='float32', )

        if not os.path.exists(destination):
            os.makedirs(destination)

        for i in range(num_images):
            image = source_data[i,:].reshape([40,40])
            label = np.where(source_labels[i] == 1)[0][0]
            dImage, label = DiffractionGenerator.generateImage(destination, i, image, label, save=True, dx=dx, dy=dy, z=z, lmbda=lmbda)
            data[i, :] = dImage.reshape(dimage_size)
        # labels haven't changed...
        dir = os.path.dirname(os.path.dirname(destination))
        np.savez(os.path.join(dir,npzName + '.npz'), data=data, labels=source_labels)
        return data, source_labels

    @classmethod
    def diffractDS1Dataset(cls):
        # bf_image, label = BrightfieldGenerator.generateImage(destination,seq,save=save)
        # load source dataset
        data_folder = '../../data/ds1-pristine/'
        train_destination = '../../data/ds2-diffraction/training/'
        test_destination = '../../data/ds2-diffraction/test/'

        train_npz = np.load(data_folder + 'training.npz')
        test_npz = np.load(data_folder + 'test.npz')

        train_data, train_labels = train_npz['data'], train_npz['labels']
        test_data, test_labels = test_npz['data'], test_npz['labels']

        #train_data, train_labels = BrightfieldGenerator.loadData(data_folder + "training/*.png")
        #test_data, test_labels = BrightfieldGenerator.loadData(data_folder + "test/*.png")
        DiffractionGenerator.generateDiffractionDataset(train_data, train_labels, train_destination, 'training')
        DiffractionGenerator.generateDiffractionDataset(test_data, test_labels, test_destination, 'test')



#DiffractionGenerator.generateNewImagePair(".", 1)
DiffractionGenerator.diffractDS1Dataset()