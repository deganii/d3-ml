import numpy as np
import os
model_folder = '../../models/'
import matplotlib.pyplot as plt
import matplotlib

class ComponentVisualizer(object):
    @classmethod
    def merge_data_series(cls, models, destination):
        merged = {}
        for model in models:
            perf_file = os.path.join(model_folder,model, 'performance.npz')
            perf_data = np.load(perf_file)
            for name,value in perf_data.items():
                if name in merged.keys():
                    # scale the time and iterations by the last value
                    if name in ['step_dt', 'iter_dt', 'time_dt']:
                        last_val = merged[name][-15]
                        value = value + (last_val - merged[name][0])
                    merged[name] = np.concatenate((merged[name][0:15], value), axis=0)
                else:
                    merged[name] = value
        cls.plot_data_series(merged)

    @classmethod
    def plot_data_series(cls, merged):
        # plot and save to disk
        iter_per_epoch = 6000
        fig = plt.figure(figsize=(4,2))
        #title = "Learning Rate"
        fig.suptitle('', fontsize=10, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        ax.plot(merged['iter_dt']/iter_per_epoch, merged['train_acc_dt'], label='Train', linewidth=0.5)
        ax.plot(merged['iter_dt'] / iter_per_epoch, merged['test_acc_dt'], label='Test', linewidth=0.5)
        ax.legend()
        fig.subplots_adjust(left = 0.17)
        fig.subplots_adjust(bottom = 0.27)
        fig.subplots_adjust(right = 0.96)
        fig.subplots_adjust(top = 0.94)
        fig.canvas.draw()
        plt.show()

# ComponentVisualizer.merge_data_series([
#     'm1-cnn-pristine',
#     'm2-cnn-pristine-double',
#     'm3-cnn-pristine-10-epoch',
#     'm4-cnn-pristine-13-epochs',
#     'm5-cnn-pristine-16-epochs',
#     'm6-cnn-pristine-20-epochs',
#     'm7-cnn-pristine-23-epochs'], '../../figures/Fig3.png')

ComponentVisualizer.merge_data_series([
    'm-14-cnn-diffraction-10-conv3',
    'm-17-cnn-diffraction-10-conv3-adam-clip'], '../../figures/Fig-DiffractionFail70.png')
