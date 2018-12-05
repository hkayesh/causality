import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


class Visualizer(object):
    def __init__(self):
        self.data_file_path = None
        self.output_dir = 'files/charts/'

    def check_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def set_data_file_path(self, path):
        self.data_file_path = path
    def set_output_dir(self, path):
        self.output_dir = path

    def read_result_data(self):
        data = None
        if os.path.exists(self.data_file_path):
            with open(self.data_file_path) as json_file:
                content = json_file.read()
                if len(content) > 0:
                    data = json.loads(content)

        return data

    def display_linechart_wiki_and_extension(self):
        labels = {
            'cnet_wiki_exp_0': '0-word',
            'cnet_wiki_exp_1': '1-word',
            'cnet_wiki_exp_2': '2-word',
            'cnet_wiki_exp_3': '3-word',
            'cnet_wiki_exp_4': '4-word',
            'cnet_wiki_exp_5': '5-word'
        }
        data = self.read_result_data()

        if data is None:
            print("No data to plot")
            return

        x_axis = list(labels.values())
        y_axis_str = [data[label]['scores'] for label in labels]

        plt.clf()
        y_axis = []
        for score_item in y_axis_str:
            row = [float(num) for num in score_item]
            y_axis.append(row)
        y_axis_ar = np.array(y_axis)

        plt.plot(x_axis, y_axis_ar[:, 0], '-*', linewidth=1, alpha=0.9, label='Accuracy')
        plt.plot(x_axis, y_axis_ar[:, 1], '-D', linewidth=1, alpha=0.9, label='Precision')
        plt.plot(x_axis, y_axis_ar[:, 2], '->', linewidth=1, alpha=0.9, label='Recall')
        plt.plot(x_axis, y_axis_ar[:, 3], '-<', linewidth=1, alpha=0.9, label='F1-score')

        plt.xlabel('Number of words')
        plt.ylabel('Scores(%)')
        plt.legend()

        self.check_output_dir()
        plt.savefig(self.output_dir + 'evaluation_scores.pdf')

    def display_linechart_news_and_extension(self):
        labels = {
            'cnet_news_exp_0': '0-word',
            'cnet_news_exp_1': '1-word',
            'cnet_news_exp_2': '2-word',
            'cnet_news_exp_3': '3-word',
            'cnet_news_exp_4': '4-word',
            'cnet_news_exp_5': '5-word'
        }
        data = self.read_result_data()

        if data is None:
            print("No data to plot")
            return

        x_axis = list(labels.values())
        y_axis_str = [data[label]['scores'] for label in labels]

        plt.clf()
        y_axis = []
        for score_item in y_axis_str:
            row = [float(num) for num in score_item]
            y_axis.append(row)
        y_axis_ar = np.array(y_axis)

        plt.plot(x_axis, y_axis_ar[:, 0], '-*', linewidth=1, alpha=0.9, label='Accuracy')
        plt.plot(x_axis, y_axis_ar[:, 1], '-D', linewidth=1, alpha=0.9, label='Precision')
        plt.plot(x_axis, y_axis_ar[:, 2], '->', linewidth=1, alpha=0.9, label='Recall')
        plt.plot(x_axis, y_axis_ar[:, 3], '-<', linewidth=1, alpha=0.9, label='F1-score')

        plt.xlabel('Number of words')
        plt.ylabel('Scores(%)')
        plt.legend()

        self.check_output_dir()
        plt.savefig(self.output_dir + 'evaluation_scores_news.pdf')

    def display_roc_curves_wiki_and_extension(self):
        result_keys = {
            'cnet_wiki_exp_0': '0-word',
            'cnet_wiki_exp_1': '1-word',
            'cnet_wiki_exp_2': '2-word',
            'cnet_wiki_exp_3': '3-word',
            'cnet_wiki_exp_4': '4-word',
            'cnet_wiki_exp_5': '5-word'
        }

        data = self.read_result_data()

        if data is None:
            print("No data to plot")
            return

        plt.clf()
        for result_key in result_keys.keys():
            mean_tpr = data[result_key]['roc']['mean_tpr']
            mean_fpr = data[result_key]['roc']['mean_fpr']

            mean_auc = auc(mean_fpr, mean_tpr)

            label = result_keys[result_key] + ' (AUC = %0.2f)' % mean_auc

            plt.plot(mean_fpr, mean_tpr, label=label, linewidth=1, alpha=.9)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()

        self.check_output_dir()
        plt.savefig(self.output_dir + 'roc_curve.pdf')

    def display_roc_curves_news_and_extension(self):
        result_keys = {
            'cnet_news_exp_0': '0-word',
            'cnet_news_exp_1': '1-word',
            'cnet_news_exp_2': '2-word',
            'cnet_news_exp_3': '3-word',
            'cnet_news_exp_4': '4-word',
            'cnet_news_exp_5': '5-word'
        }

        data = self.read_result_data()

        if data is None:
            print("No data to plot")
            return

        plt.clf()
        for result_key in result_keys.keys():
            mean_tpr = data[result_key]['roc']['mean_tpr']
            mean_fpr = data[result_key]['roc']['mean_fpr']

            mean_auc = auc(mean_fpr, mean_tpr)

            label = result_keys[result_key] + ' (AUC = %0.2f)' % mean_auc

            plt.plot(mean_fpr, mean_tpr, label=label, linewidth=1, alpha=.9)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()

        self.check_output_dir()
        plt.savefig(self.output_dir + 'roc_curve_news.pdf')
