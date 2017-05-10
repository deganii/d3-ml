
from __future__ import print_function
import PIL.ImageOps
from PIL import Image, ImageDraw
import random
import numpy as np
import os
import ntpath

from scipy import misc
import glob
# generator of labeled bright-field test images

class BrightfieldGenerator(object):

    @classmethod
    def generateImage(cls, destination, seq, save = True, draw_mask = False, invert = True):
        # create at double the size, then downsample
        image = Image.new('L', (40, 40))
        draw = ImageDraw.Draw(image)
        if draw_mask:
            mask = Image.new('L', (40, 40))
            mask_draw = ImageDraw.Draw(mask)

        #  3 pixels of stoachastic behavior in cell size
        s = 2
        t = 1
        sh = 3
        sx, sy = (random.randint(-s,s), (random.randint(-s,s)))
        tx ,ty = (random.randint(-t,t), (random.randint(-t,t)))
        shx, shy = (random.randint(-sh, sh), (random.randint(-sh, sh)))

        bounds = [12 + tx - sx, 12+ty - sy, 27+tx+sx, 27+ty+sy]
        BrightfieldGenerator.draw_ellipse(image, bounds, width=2, outline='white')

        if mask:
            BrightfieldGenerator.draw_ellipse(mask, bounds, width=2, outline='white')
            mask_draw.ellipse(bounds, fill='white', outline=None)

        # draw beads
        cell_a, cell_b = (bounds[2] - bounds[0]) / 2, (bounds[3] - bounds[1] )/ 2
        cell_x, cell_y = bounds[0] + cell_a, bounds[1] + cell_b
        #draw_ellipse(image, [cell_x-1, cell_y-1, cell_x + 1, cell_y+1], width=1, outline='white')
        num_beads = random.choice([0,0,0,0,1,1,2,3,4,4,5,5])
        #print("Num beads: {0}".format(num_beads))
        bead_radius = 3

        # this is the distance from the cell
        # to the center of the bead
        cell_a = cell_a + bead_radius
        cell_b = cell_b + bead_radius
        #num_beads = 5

        # choose non-overlapping binding sites
        bind_sites = []
        min_separation = 30
        for i in range(num_beads):
            while True:
                bind_site = random.randint(-180,180)
                if not any( angle_dist(bind_site,b) < min_separation for b in bind_sites):
                    bind_sites.append(bind_site)
                    break
            eps = 1e-9
            bead_xrel = (cell_a*cell_b) / np.sqrt(cell_b **2 + (cell_a ** 2) * (eps + np.tan(np.deg2rad(bind_site))**2))
            bead_yrel = (cell_a * cell_b) / np.sqrt(cell_a ** 2 + (cell_b ** 2) / (eps + np.tan(np.deg2rad(bind_site))**2))
            if bind_site > 90 or bind_site < -90:
                bead_xrel = bead_xrel * -1
            if bind_site < 0:
                bead_yrel = bead_yrel * -1
            bead_xabs = bead_xrel +  cell_x
            bead_yabs = bead_yrel +  cell_y
            bead_bound = [bead_xabs - bead_radius, bead_yabs - bead_radius, bead_xabs + bead_radius, bead_yabs + bead_radius]
            BrightfieldGenerator.draw_ellipse(image, bead_bound, width=2, outline='white')

            if mask:
                BrightfieldGenerator.draw_ellipse(mask, bead_bound, width=2, outline='white')
                mask_draw.ellipse(bead_bound, fill='white', outline=None)

        #num_stray_beads = random.choice([0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5])
        #stray_beads = []
        #for i in range(num_stray_beads):
        #    stray_bead = ()


        global_rotate = random.randint(0,180)
        image = image.rotate(global_rotate, resample=Image.BILINEAR)
        if mask:
            mask = mask.rotate(global_rotate, resample=Image.BILINEAR)
        #image = image.resize((40,40), resample=Image.LANCZOS)
        # image = image.transform(
        #     image.size, Image.AFFINE,
        #     [
        #         sx, shx, 0,
        #         shy, sy, 0,
        #     ],
        #     Image.BILINEAR
        # )
        if invert:
            image = PIL.ImageOps.invert(image)

        if save:
            if not os.path.exists(destination):
                os.makedirs(destination)
            image.save(os.path.join(destination, '{0:05}-{1}.png'.format(seq, num_beads )))
            if mask:
                image.save(os.path.join(destination, '{0:05}-{1}-M.png'.format(seq, num_beads)))

        if mask:
            return image, num_beads, mask
        return image, num_beads

    @classmethod
    def makenpz(cls, path, dest_name):
        dir = os.path.dirname(os.path.dirname(path))
        data, labels = BrightfieldGenerator.loadData(path)
        np.savez(os.path.join(dir,dest_name + '.npz'), data=data, labels=labels)

    @classmethod
    def loadData(cls, destination):
        #images = []
        files = glob.glob(destination)
        image_size = 40*40

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



    @classmethod
    def draw_ellipse(cls, image, bounds, width=1, outline='white', antialias=4):
        """Improved ellipse drawing function, based on PIL.ImageDraw."""

        # Use a single channel image (mode='L') as mask.
        # The size of the mask can be increased relative to the imput image
        # to get smoother looking results.
        mask = Image.new(
            size=[int(dim * antialias) for dim in image.size],
            mode='L', color='black')
        draw = ImageDraw.Draw(mask)

        # draw outer shape in white (color) and inner shape in black (transparent)
        for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
            left, top = [(value + offset) * antialias for value in bounds[:2]]
            right, bottom = [(value - offset) * antialias for value in bounds[2:]]
            draw.ellipse([left, top, right, bottom], fill=fill)

        # downsample the mask using PIL.Image.LANCZOS
        # (a high-quality downsampling filter).
        mask = mask.resize(image.size, Image.LANCZOS)
        # paste outline color to input image through the mask
        image.paste(outline, mask=mask)


def angle_dist(a1, a2):
    a = a2 - a1
    if a > 180:
        a = a - 360
    if a < -180:
        a = a + 360
    return np.abs(a)

def generatePristineBatch(destination, num_images = 10000):
    for i in range(num_images):
        BrightfieldGenerator.generateImage(destination, i)
        if i % 1000 == 0:
            print("Generating {0}/{1}  -  (2)%".format(i, num_images), 100.*i / num_images)

#generatePristineBatch("../../data/ds1-pristine/training", 60000)
#d, l = BrightfieldGenerator.loadData("../../data/ds1-pristine/*.png")

#BrightfieldGenerator.makenpz("../../data/ds1-pristine/training/*.png", 'training')
#BrightfieldGenerator.makenpz("../../data/ds1-pristine/test/*.png", 'test')
