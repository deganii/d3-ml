import scipy.misc
import os
from PIL import Image
import ntpath
import glob
import numpy as np

def clean1():
    square_len = 1365
    input_folder = '../../data/ds5-real-data/original/'
    output_folder = '../../data/ds5-real-data/square/'
    for file in os.listdir(input_folder):
        #im = scipy.misc.imread(input_folder+file)
        im = Image.open(input_folder+file)
        w,h = im.size
        dest_im = Image.new('RGB', (square_len, square_len))
        dest_im.paste(im, [0,0])
        dest_name = ntpath.basename(file)
        dest_path = os.path.join(output_folder, dest_name)
        scipy.misc.imsave(dest_path, dest_im)


def makeGrayscale():
    data_folder = '../../data/ds5-real-data/square/'
    output_folder = '../../data/ds5-real-data/squaremask/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in glob.glob(data_folder + '*.png'):
        img = Image.open(file)

        img = img.convert('L')
        #img = img.point(lambda x: 0 if x < 64 else 255, '1')

        #gray = Image.new( 'L', img.size)
        #gray.paste(img, [0,0])
        #img_a = np.array(img).reshape([1365,1365])
        dest_name = ntpath.basename(file)
        dest_path = os.path.join(output_folder, dest_name)
        #scipy.misc.imsave(dest_path, img_a)
        #img.save(dest_path, gray)
        scipy.misc.imsave(dest_path, img)

def loadData():
    image_size = 200*200

    data = np.zeros([len(files),image_size], dtype='float32',)
    labels = np.zeros([len(files), 6], dtype='float32',)
    count = 0
    for image_path in files:
        image = misc.imread(image_path).reshape(image_size)
        seq_num, label = (int(d) for d in os.path.splitext(ntpath.basename(image_path))[0].split('-'))
        #np.copyto(data[seq_num*image_size, (seq_num+1)*image_size], image)
        data[seq_num, :] = image
        # one-hot encoding
        labels[seq_num, :] = np.zeros(6, dtype='float32')
        labels[seq_num, label] = 1.0
    return (data, labels)


def makeSmall():
    output_folder = '../../data/ds5-real-data/squaremask-augmented-200'
    data_folder = '../../data/ds5-real-data/squaremask/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in glob.glob(data_folder + "*.*" ):
        img = Image.open(file).convert('L')
        trans = [0,200,300, 400,500,600,700,800,965]
        rot = [0, 90, 270, 180]
        for i in trans:
            for j in trans:
                for ang in rot:
                    temp = img.crop([i, j, 200 + i, 200 + j])
                    temp = temp.rotate(ang, resample=Image.BICUBIC )

                    dest_name, dest_ext = os.path.splitext(ntpath.basename(file))
                    transformation = "_tx_{0}_ty_{1}_rot_{2}".format(i,j,ang)
                    dest_path = os.path.join(output_folder, dest_name + transformation + dest_ext)
                    scipy.misc.imsave(dest_path, temp)

        #img = img.point(lambda x: 0 if x < 64 else 255, '1')



        # #gray = Image.new( 'L', img.size)
        # #gray.paste(img, [0,0])
        # #img_a = np.array(img).reshape([1365,1365])
        # dest_name = ntpath.basename(file)
        # dest_path = os.path.join(output_folder, dest_name)
        # #scipy.misc.imsave(dest_path, img_a)
        # #img.save(dest_path, gray)
        # scipy.misc.imsave(dest_path, img)
makeSmall()