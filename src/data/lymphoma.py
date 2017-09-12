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
    def partitionTrainingAndTestSet(cls):
        data_folder = '../../data/ds7-lymphoma/'
        data_npz = np.load(data_folder + 'all.npz')

        all_data, all_labels = data_npz['data'], data_npz['labels']
        indices = np.random.permutation(all_data.shape[0])

        test_count = int(np.floor(all_data.shape[0] * 0.2))
        test_idx, training_idx = indices[:test_count], indices[test_count:]
        training_data, test_data = all_data[training_idx, :], all_data[test_idx, :]
        training_labels, test_labels = all_labels[training_idx, :, :], all_labels[test_idx, :, :]
        np.savez(os.path.join(data_folder, 'training.npz'), data=training_data, labels=training_labels)
        np.savez(os.path.join(data_folder, 'test.npz'), data=test_data, labels=test_labels)

    @classmethod
    def generateImage(cls, set_name, output_folder='../../data/ds7-lymphoma/training'):
        # open lymphoma image
        sample_data = h5py.File('../images/Daudi_PS_NAV_Postwash_image1.mat', 'r')
        # load a sample image
        subNormAmp = sample_data['subNormAmp']
        reconImage = sample_data['ReconImage']

        #reshaped = [[reconImage[m, n] for m in range(reconImage.shape[0])] for n in range(reconImage.shape[1])]
        # convert to numpy array

        subNormAmp = np.asarray(subNormAmp)
        viewHologram = Image.fromarray(np.transpose(np.uint8(255.0 * subNormAmp / np.max(subNormAmp))))

        #scipy.misc.imsave('./test_viewhologram.png', viewHologram)

        # convert from matlab tuples to a proper np array
        recon = np.asarray([[[num for num in row] for row in rows] for rows in reconImage])
        reconReal = recon[:, :, 0]
        reconImag = recon[:, :, 1]

        viewReconReal = reconReal + np.abs(np.min(reconReal))
        viewReconImag = reconImag + np.abs(np.min(reconImag))

        viewReconReal = Image.fromarray(np.transpose(np.uint8(255.0 * (viewReconReal) / np.max(viewReconReal))))
        viewReconImag = Image.fromarray(np.transpose(np.uint8(255.0 * (viewReconImag) / np.max(viewReconImag))))

        #scipy.misc.imsave('./test_viewreconReal.png', viewReconReal)
        #scipy.misc.imsave('./test_viewreconImag.png', viewReconImag)


        M = subNormAmp.shape[0]
        N = subNormAmp.shape[1]

        # tile the image
        stride = 100
        tile = [200, 200]
        seq = 0

        last_M = int(M - (M % stride))
        last_N = int(N - (N % stride))

        M_count = int(np.floor(M/stride))
        N_count = int(np.floor(N/stride))

        data = np.zeros((4 * M_count * N_count, tile[0] * tile[1]))
        labels = np.zeros((4 * M_count * N_count, 2, tile[0] * tile[1]))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for rot in range(0, 360, 90):
            for m in range(0, last_M, stride):
                for n in range(0, last_N, stride):

                    st_m, end_m, st_n, end_n = m, tile[0] + m,  n, tile[1] + n

                    if end_m >= M:
                        st_m, end_m = M - 1 - tile[0], M - 1
                    if end_n >= N:
                        st_n, end_n = N - 1 - tile[1], N - 1

                    crop_mn = [st_m, st_n, end_m, end_n]

                    holoTile = viewHologram.crop(crop_mn)
                    realTile = viewReconReal.crop(crop_mn)
                    imageTile = viewReconImag.crop(crop_mn)

                    holoTile = holoTile.rotate(rot, resample=Image.BICUBIC)
                    realTile = realTile.rotate(rot, resample=Image.BICUBIC)
                    imageTile = imageTile.rotate(rot, resample=Image.BICUBIC)

                    transformation = "tm_{0}_tn_{1}_rot_{2}".format(st_m, st_n, rot)

                    holoDestFilename = '{0:05}-H-{1}.png'.format(seq,transformation)
                    realDestFilename = '{0:05}-R-{1}.png'.format(seq,transformation)
                    imagDestFilename = '{0:05}-I-{1}.png'.format(seq,transformation)

                    scipy.misc.imsave(os.path.join(output_folder, holoDestFilename), holoTile)
                    scipy.misc.imsave(os.path.join(output_folder, realDestFilename), realTile)
                    scipy.misc.imsave(os.path.join(output_folder, imagDestFilename), imageTile)

                    # append the raw data to the
                    data[seq, :] = np.rot90(subNormAmp[st_m:end_m, st_n:end_n], int(rot / 90)).reshape(tile[0] * tile[1])
                    labels[seq, 0, :] = np.rot90(reconReal[st_m:end_m, st_n:end_n], int(rot / 90)).reshape(tile[0] * tile[1])
                    labels[seq, 1, :] = np.rot90(reconImag[st_m:end_m, st_n:end_n], int(rot / 90)).reshape(tile[0] * tile[1])

                    seq = seq + 1


        dir_name = os.path.dirname(output_folder)



        np.savez(os.path.join(dir_name, set_name + '.npz'), data=data, labels=labels)




#LymphomaGenerator.generateImage('all')
LymphomaGenerator.partitionTrainingAndTestSet()