from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet.image_util import BaseDataProvider
import numpy as np
import os
import glob
import scipy.misc
import ntpath

data_folder = '../../data/ds7-lymphoma/'

train_npz = np.load(data_folder + 'training.npz')
train_data, train_labels = train_npz['data'], train_npz['labels']

class EnsembleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 256
    idx = 0

    def __init__(self, **kwargs):
        super(EnsembleDataProvider, self).__init__()
        self.nx = 200
        self.ny = 200
        self.kwargs = kwargs

    def _next_data(self):
        data = train_data[self.idx % train_data.shape[0],:].reshape((self.nx, self.ny,1))
        label = train_labels[self.idx % train_labels.shape[0],:].reshape((self.nx, self.ny,2))
        #label = label == 255
        self.idx = self.idx + 1
        return data, label

model_folder = '../../models/m-28-unet-lymphoma-200/'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

generator = EnsembleDataProvider()
net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=8, features_root=32)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, model_folder + "unet_trained", training_iters=20, epochs=50, display_step=2, restore=False)

