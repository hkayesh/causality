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
    dataset_file = 'causal_pairs_dataset.csv'
    result_file = 'results.txt'
    evaluation.run_experiment(dataset_file=dataset_file, result_file=result_file, n_pair=1000, n_expand=5)


    ## data visualization
    # visualizer.set_data_file_path(result_file)
    # visualizer.display_line_chart()
    # visualizer.display_roc_curves()


