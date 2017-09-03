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
# generator of labeled lymphoma test images
import h5py

class LymphomaGenerator(object):
    @classmethod
    def generateImage(cls, set_name, output_folder='../../data/ds7-lymphoma/training'):
        # open lymphoma image
        sample_data = h5py.File('../images/Daudi_PS_NAV_Postwash_image1.mat', 'r')
        # load a sample image
        subNormAmp = sample_data['subNormAmp']
        reconImage = sample_data['ReconImage']
        M = subNormAmp.shape[0]
        N = subNormAmp.shape[1]
        #reshaped = [[reconImage[m, n] for m in range(reconImage.shape[0])] for n in range(reconImage.shape[1])]
        # convert to numpy array

        subNormAmpNp = np.transpose(np.asarray(subNormAmp))
        viewHologram = Image.fromarray(np.uint8(255.0 *  subNormAmpNp / np.max(subNormAmpNp)))
        #scipy.misc.imsave('./test_viewhologram.png', viewHologram)

        # convert from matlab tuples to a proper np array
        recon = np.asarray([[[num for num in row] for row in rows] for rows in reconImage])
        reconReal = np.transpose(recon[:, :, 0])
        reconImag = np.transpose(recon[:, :, 1])

        viewReconReal = reconReal + np.abs(np.min(reconReal))
        viewReconImag = reconImag + np.abs(np.min(reconImag))

        viewReconReal = Image.fromarray(np.uint8(255.0 * (viewReconReal) / np.max(viewReconReal)))
        viewReconImag = Image.fromarray(np.uint8(255.0 * (viewReconImag) / np.max(viewReconImag)))

        #scipy.misc.imsave('./test_viewreconReal.png', viewReconReal)
        #scipy.misc.imsave('./test_viewreconImag.png', viewReconImag)

        # tile the image
        stride = 25
        tile = [200, 200]
        seq = 0

        last_M = int(M - (M % stride) - tile[0])
        last_N = int(N - (N % stride) - tile[1])

        M_count = int(np.floor((M-tile[0])/stride))
        N_count = int(np.floor((N-tile[1])/stride))

        data = np.zeros((4 * M_count * N_count, tile[0] * tile[1]))
        labels = np.zeros((4 * M_count * N_count, 2, tile[0] * tile[1]))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for rot in range(0, 360, 90):
            for i in range(0, last_M, stride):
                for j in range(0, last_N, stride):
                    holoTile = viewHologram.crop([i, j, tile[0] + i, tile[1] + j])
                    realTile = viewReconReal.crop([i, j, tile[0] + i, tile[1] + j])
                    imageTile = viewReconImag.crop([i, j, tile[0] + i, tile[1] + j])

                    holoTile = holoTile.rotate(rot, resample=Image.BICUBIC)
                    realTile = realTile.rotate(rot, resample=Image.BICUBIC)
                    imageTile = imageTile.rotate(rot, resample=Image.BICUBIC)

                    transformation = "_tx_{0}_ty_{1}_rot_{2}".format(i, j, rot)

                    holoDestFilename = '{0:05}-H-{1}.png'.format(seq,transformation)
                    realDestFilename = '{0:05}-R-{1}.png'.format(seq,transformation)
                    imagDestFilename = '{0:05}-I-{1}.png'.format(seq,transformation)

                    scipy.misc.imsave(os.path.join(output_folder, holoDestFilename), holoTile)
                    scipy.misc.imsave(os.path.join(output_folder, realDestFilename), realTile)
                    scipy.misc.imsave(os.path.join(output_folder, imagDestFilename), imageTile)

                    # append the raw data to the
                    data[seq, :] = subNormAmp[i:tile[0]+i, j:tile[1]+j].reshape(tile[0] * tile[1])
                    labels[seq, 0, :] = reconReal[i:tile[0] + i, j:tile[1] + j].reshape(tile[0] * tile[1])
                    labels[seq, 1, :] = reconImag[i:tile[0] + i, j:tile[1] + j].reshape(tile[0] * tile[1])
                    seq = seq + 1


        dir_name = os.path.dirname(output_folder)



        np.savez(os.path.join(dir_name, set_name + '.npz'), data=data, labels=labels)


LymphomaGenerator.generateImage('training')