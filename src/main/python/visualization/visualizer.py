import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


class Visualizer(object):
    def __init__(self):
        self.data_file_path = None

    def set_data_file_path(self, path):
        self.data_file_path = path

    def read_result_data(self):
        data = None
        if os.path.exists(self.data_file_path):
            with open(self.data_file_path) as json_file:
                content = json_file.read()
                if len(content) > 0:
                    data = json.loads(content)

        return data

    def display_line_chart(self):
        data = self.read_result_data()

        if data is None:
            print("No data to plot")
            return

        plot_data = data['scores']

        x_axis = list(plot_data.keys())
        y_axis = []

        plt.clf()
        for label in x_axis:
            row = [float(num) for num in plot_data[label]]
            y_axis.append(row)
        y_axis_ar = np.array(y_axis)

        plt.plot(x_axis, y_axis_ar[:, 0], '-*', linewidth=1, alpha=0.9, label='Accuracy')
        plt.plot(x_axis, y_axis_ar[:, 1], '-D', linewidth=1, alpha=0.9, label='Precision')
        plt.plot(x_axis, y_axis_ar[:, 2], '->', linewidth=1, alpha=0.9, label='Recall')
        plt.plot(x_axis, y_axis_ar[:, 3], '-<', linewidth=1, alpha=0.9, label='F1-score')

        plt.xlabel('Number of words')
        plt.ylabel('Scores(%)')
        plt.legend()

        plt.show()

    def display_roc_curves(self):
        data = self.read_result_data()

        if data is None:
            print("No data to plot")
            return

        plot_data = data['roc']
        labels = list(plot_data.keys())

        for label in labels:
            mean_tpr = plot_data[label]['mean_tpr']
            mean_fpr = plot_data[label]['mean_fpr']

            mean_auc = auc(mean_fpr, mean_tpr)

            label = label + ' (AUC = %0.2f)' % mean_auc

            plt.plot(mean_fpr, mean_tpr, label=label, linewidth=1, alpha=.9)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()
