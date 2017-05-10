from data import DiffractionGenerator
from data import BrightfieldGenerator
import PIL.ImageOps
from PIL import Image, ImageDraw
import random
import numpy as np
import os
import scipy.misc

class EnsembleGenerator(object):
    # build a 400x300 image with 25 cells, beads, and noise
    @classmethod
    def generateEnsemble(self, destination, seq, width = 300, height = 200, n_cells = 10, n_beads = 30, save=True):
        image = Image.new('L', (width, height), color=0)
        mask = Image.new('L', (width, height), color=0)
        bead_radius = 3
        centers = []
        bead_centers = []
        maxTries = 100
        for i in range(n_cells):
            img, num_beads, imask = BrightfieldGenerator.generateImage(None, i, save=False, draw_mask=True, invert=False)
            for j in range(maxTries):
                loc = [random.randint(0,width - img.width), random.randint(0,height - img.height)]
                cen = np.asarray([loc[0] + img.width/2, loc[1] + img.height / 2])
                if not any([  np.linalg.norm(cen - other)  < 50  for other in centers]):
                    centers.append(cen)
                    image.paste(img, loc)
                    mask.paste(imask,loc)
                    break
        # add stray beads
        for i in range(n_beads):
            for j in range(maxTries):
                loc = [random.randint(0, width), random.randint(0, height)]
                cen = np.asarray([loc[0] + bead_radius / 2, loc[1] + bead_radius / 2])
                bead_collision = any([np.linalg.norm(cen - other) < 10 for other in bead_centers])
                cell_collision = any([np.linalg.norm(cen - other) < 35 for other in centers])
                if not (bead_collision or cell_collision):
                    bead_centers.append(cen)
                    bead_bound = [loc[0], loc[1], loc[0] + 2*bead_radius, loc[1] + 2*bead_radius]
                    BrightfieldGenerator.draw_ellipse(image, bead_bound, width=2, outline='white')
                    break

        image = PIL.ImageOps.invert(image)
        if save:
            if not os.path.exists(destination):
                os.makedirs(destination)
            image.save(os.path.join(destination, '{0:05}.png'.format(seq)))
            mask.save(os.path.join(destination, '{0:05}-M.png'.format(seq)))
        return image, mask

def generateEnsembleBatch(destination, npzName, num_images = 10000,  width = 300, height = 200, ):
    image_size = width * height
    if not os.path.exists(destination):
        os.makedirs(destination)

    data = np.zeros([num_images, image_size], dtype='float32', )
    labels = np.zeros([num_images, image_size], dtype='float32', )
    for i in range(num_images):
        image, mask = EnsembleGenerator.generateEnsemble(destination, i, width = width, n_cells=7 )
        data[i, :] = np.array(image).reshape(image_size)
        labels[i, :] = np.array(mask).reshape(image_size)
        if i % 1000 == 0:
            print("Generating {0}/{1}  -  (2)%".format(i, num_images), 100.*i / num_images)
    dir = os.path.dirname(os.path.dirname(destination))
    np.savez(os.path.join(dir,npzName + '.npz'), data=data, labels=labels)
    return data, labels

def diffractEnsembleBatch(destination,  width = 200, height = 200, ):
    # bf_image, label = BrightfieldGenerator.generateImage(destination,seq,save=save)
    # load source dataset
    data_folder = '../../data/ds3-ensemble-square/'

    train_npz = np.load(data_folder + 'training.npz')

    train_data, train_labels = train_npz['data'], train_npz['labels']

    num_images = train_data.shape[0]
    dimage_size = width*height
    data = np.zeros([num_images, dimage_size], dtype='float32', )

    if not os.path.exists(destination):
        os.makedirs(destination)

    for i in range(num_images):
        image = train_data[i, :].reshape([width,height])
        mask = train_labels[i, :].reshape([width,height])
        destfile= 'D{0:05}.png'.format(i)
        dImage, _ = DiffractionGenerator.generateImage(destination, i, image, None, save=True, pad_width=0, destfilename=destfile)
        data[i, :] = dImage.reshape(dimage_size)
        #write the mask
        maskfile = 'D{0:05}-M.png'.format(i)
        scipy.misc.imsave(os.path.join(destination, maskfile), mask)

    # labels haven't changed...
    dir = os.path.dirname(os.path.dirname(destination))
    np.savez(os.path.join(dir, 'training.npz'), data=data, labels=train_labels)
    return data, train_labels

#generateEnsembleBatch('../../data/ds3-ensemble-square/training/', 'training', num_images=2000, width=200 )
#generateEnsembleBatch('../../data/ds3-ensemble/test/', 'test', num_images=500, width=200)

train_destination = '../../data/ds4-ensemble-diffraction/training/'
diffractEnsembleBatch(train_destination, width=200, height=200)
