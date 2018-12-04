import numpy as np
np.random.seed(1) # NumPy
import random
random.seed(2) # Python
from tensorflow import set_random_seed
set_random_seed(3) # Tensorflow

from causality_detection.feed_forward import Evaluation
from visualization.visualizer import Visualizer


if __name__ == '__main__':
    evaluation = Evaluation()
    visualizer = Visualizer()

    experiment_key = 'cnet_wiki_exp_0'

    settings = {
        'dataset_file': 'causal_pairs_dataset.csv',
        'result_file': 'results.json',
        'embedding_model_file': 'files/GoogleNews-vectors-negative300.bin',
        'causal_net_file': 'causal_net_1m.pickle',
        'n_pair': 1000,
        'n_expand': 0,
        'result_key': 'cnet_wiki_exp_0'
    }

    # evaluation.run_experiment(settings=settings)

    settings['threshold'] = 10
    settings['result_key'] = 'luo_threshold_10'
    # evaluation.run_experiment_on_luos_method(settings)


    ## data visualization
    visualizer.set_data_file_path(settings['result_file'])
    visualizer.set_output_dir(path='files/charts/')

    visualizer.display_linechart_wiki_and_extension()
    visualizer.display_roc_curves_wiki_and_extension()

    visualizer.display_linechart_news_and_extension()
    visualizer.display_roc_curves_news_and_extension()



