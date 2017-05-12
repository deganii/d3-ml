import matplotlib.pyplot as plt
import math
from PIL import Image
import glob
import os
import numpy as np
import ntpath
import scipy.misc

class ComponentVisualizer(object):
    @classmethod
    def plot_filter(cls, units):
        filters = units.shape[3]
        plt.figure()
        n_columns = 3
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            #plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.show()

    @classmethod
    def saveTiledFilterOutputs(cls, units, destination, train_data, labels, n_columns=4, cropx=0, cropy = 0, ):
        filters = units.shape[3]
        tiledImages = []
        items_to_output = 5
        for i in range(items_to_output):
            images = [scipy.misc.toimage(units[i,:,:,f].reshape([units.shape[1], units.shape[2]]), mode='L') for f in range(filters)]
            src_image = scipy.misc.toimage(train_data[i, :].reshape(40,40))
            if src_image.width > units.shape[1]:
                src_image = src_image.resize((units.shape[1], units.shape[2]), resample=Image.BICUBIC)
            for j in reversed(range(int(filters/n_columns))):
                images.insert( j *  n_columns, src_image)
            label = np.where(labels[i] == 1)[0][0]
            tiledImages.append(ComponentVisualizer.saveTiledImages(images, destination.format(i,label), n_columns=n_columns +1, cropx=cropx, cropy = cropy))
        # now tile the tiles!
        c_tileh = units.shape[2] - 2 * (cropy + 2)
        imageCompoundTiledImage = Image.new('L', (tiledImages[0].width, (items_to_output-2)*c_tileh), color=255)
        tiledImages.pop(3)
        tiledImages.pop(2)
        for idx, tiledImage in enumerate(tiledImages):
            #cropped = tiledImage.crop([0,0,tiledImage.width, items_to_output*(c_tileh)])
            imageCompoundTiledImage.paste(tiledImage, [0, idx * c_tileh ])
        imageCompoundTiledImage.save(destination.format('C', 4))




    @classmethod
    def saveTiledFilters(cls, units, destination, n_columns=4, cropx=0, cropy = 0):
        filters = units.shape[3]
        images = [scipy.misc.toimage(units[:,:,0, f].reshape([units.shape[0], units.shape[1]]), mode='L') for f in range(filters)]
        ComponentVisualizer.saveTiledImages(images, destination, n_columns=n_columns, cropx=cropx, cropy = cropy)


    @classmethod
    def saveTiledImages(cls, files, destination, n_columns=4, cropx=0, cropy = 0):
        if isinstance(files[0],str):
            files = [Image.open(f) for f in files]

        width, height = files[0].size
        width, height = width - 2*cropx, height - 2*cropy
        n_rows = int((len(files))/n_columns)

        a_height = int(height * n_rows)
        a_width = int(width * n_columns)
        image = Image.new('L', (a_width, a_height), color=255)

        for row in range(n_rows):
            for col in range(n_columns):
                y0 = row * height - cropy
                x0 = col * width - cropx
                #print('Accessing: {0}, {1}. {2}', row, col, row*n_columns+col)
                tile = files[row*n_columns+col]
                image.paste(tile, (x0,y0))
        image.save(destination)
        # send back the tiled img
        return image

    @classmethod
    def saveTiledUnet(cls, n_columns=4, cropx=0, cropy = 0):
        files = glob.glob('../../figures/diffracted-predictions/*.*')

        files = [Image.open(f).crop((320,0,320+160,160)) for f in files]
        cls.saveTiledImages(files, '../../figures/unet-tiled-diffraction.png', n_columns=n_columns)




ComponentVisualizer.saveTiledUnet()
# diffPrefix = '../../data/ds2-diffraction/training/D'
# files = glob.glob('../../data/ds1-pristine/training/000*.png')[0:5]
# dfiles = [diffPrefix + ntpath.basename(f) for f in files ]
# ncols = 4
# ComponentVisualizer.saveTiledImages(files, '../../figures/fig2-undiff4.png', numcols=ncols, cropx=-20, cropy=-20)
# ComponentVisualizer.saveTiledImages(dfiles, '../../figures/fig2-diff4.png',  numcols=ncols, cropx=-0, cropy=-0)

