
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
    def generateImage(cls, destination, seq, bf_image, label, save = True, dx = 0.8,  dy = 0.8, z = 0.2, lmbda = 405):
        U0 = np.array(bf_image)
        U0 = np.pad(U0, pad_width=30, mode='constant', constant_values=255 )

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
    def generateDiffractionDataset(cls, destination, seq, save=True, dx=0.8, dy=0.8, z=0.2, lmbda=405):
        bf_image, label = BrightfieldGenerator.generateImage(destination,seq,save=save)
        return DiffractionGenerator.generateImage( destination, seq, bf_image, label, save=save, dx=dx, dy=dy, z=z, lmbda=lmbda)


DiffractionGenerator.generateNewImagePair(".", 1)