from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet.image_util import BaseDataProvider
import numpy as np
import os
import glob
import scipy.misc
import ntpath
# data_folder = '../../data/ds3-ensemble-square/'
# data_folder = '../../data/ds4-ensemble-diffraction/'
data_folder = '../../data/ds5-real-data/square/'
mask_folder = '../../data/ds5-real-data/squaremask-small/'
#train_npz = np.load(data_folder + 'training.npz')
#test_npz = np.load(data_folder + 'test.npz')

#train_data, train_labels = train_npz['data'], train_npz['labels']
#test_data, test_labels = test_npz['data'], test_npz['labels']

class EnsembleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    idx = 0

    def __init__(self, **kwargs):
        super(EnsembleDataProvider, self).__init__()
        self.nx = 200
        self.ny = 200
        self.kwargs = kwargs

    def _next_data(self):
        data = train_data[self.idx % train_data.shape[0],:].reshape((self.nx, self.ny,1))
        label = train_labels[self.idx % train_labels.shape[0],:].reshape((self.nx, self.ny))
        label = label == 255
        self.idx = self.idx + 1
        return data, label

class RealDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    idx = 0
    files = glob.glob(mask_folder + "*.png" )
    #im = scipy.misc.imread(input_folder+file)

    def __init__(self, **kwargs):
        super(RealDataProvider, self).__init__()
        self.nx = 200
        self.ny = 200
        self.kwargs = kwargs

    def _next_data(self):
        datafile = self.files[self.idx]
        data = scipy.misc.imread(datafile) #.reshape((self.nx, self.ny, 1))

        basename =  ntpath.basename(datafile)
        labelname = basename[0:basename.find('.mat')] + '_composite.tif'
        label = scipy.misc.imread(mask_folder + labelname)
        label = label == 255
        self.idx = (self.idx + 1) % len(self.files)
        return data, label


    #def _process_labels(self, label):
    #    return label


#model_folder = '../../models/m23-unet-diffraction-full/'
model_folder = '../../models/m-24-unet-real-data-augmented/'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

#generator = EnsembleDataProvider()
generator = RealDataProvider()
net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, model_folder + "unet_trained", training_iters=2000, epochs=20, display_step=2, restore=True)

