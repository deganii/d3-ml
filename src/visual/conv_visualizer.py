import matplotlib.pyplot as plt
import math

class ConvVisualizer(object):
    @classmethod
    def plot_filter(cls, units):
        filters = units.shape[3]
        plt.figure(1, figsize=(40,40))
        n_columns = 3
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            #plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.show()


