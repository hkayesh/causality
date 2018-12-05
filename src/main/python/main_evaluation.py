import argparse
import numpy as np

np.random.seed(1)  # NumPy
import random

random.seed(2)  # Python
from tensorflow import set_random_seed

set_random_seed(3)  # Tensorflow

from causality_detection.feed_forward import Evaluation
from visualization.visualizer import Visualizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", default=None, help="Experiment key")
    parser.add_argument("-v", "--visualize", default='no', help="Generate output charts")

    args = parser.parse_args()
    experiment_key = args.key
    visualize = args.visualize

    evaluation = Evaluation()
    visualizer = Visualizer()

    settings = {
        'dataset_file': 'causal_pairs_dataset_1000.csv',
        'result_file': 'results.json',
        'embedding_model_file': 'files/GoogleNews-vectors-negative300.bin',
        'causal_net_file': 'causal_net_1m.pickle',
        'n_pair': 1000,
        'n_expand': 0,
        'result_key': 'cnet_wiki_exp_0'
    }

    if experiment_key == 'cnet_wiki_exp_0':
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_wiki_exp_1':
        settings['n_expand'] = 1
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_wiki_exp_2':
        settings['n_expand'] = 2
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_wiki_exp_3':
        settings['n_expand'] = 3
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_wiki_exp_4':
        settings['n_expand'] = 4
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_wiki_exp_5':
        settings['n_expand'] = 5
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)


    # News article causal net
    settings['causal_net_file'] = 'causal_net_news.pickle'

    if experiment_key == 'cnet_news_exp_0':
        settings['n_expand'] = 0
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_news_exp_1':
        settings['n_expand'] = 1
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_news_exp_2':
        settings['n_expand'] = 2
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_news_exp_3':
        settings['n_expand'] = 3
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_news_exp_4':
        settings['n_expand'] = 4
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'cnet_news_exp_5':
        settings['n_expand'] = 5
        settings['result_key'] = experiment_key
        evaluation.run_experiment(settings=settings)

    if experiment_key == 'luo_threshold_10':
        settings['threshold'] = 10
        settings['result_key'] = experiment_key
        evaluation.run_experiment_on_luos_method(settings)

    ## data visualization
    if visualize == 'yes':
        visualizer.set_data_file_path(settings['result_file'])
        visualizer.set_output_dir(path='files/charts/')

        visualizer.display_linechart_wiki_and_extension()
        visualizer.display_roc_curves_wiki_and_extension()

        visualizer.display_linechart_news_and_extension()
        visualizer.display_roc_curves_news_and_extension()



