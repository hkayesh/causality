import numpy as np
np.random.seed(1) # NumPy
import random
random.seed(2) # Python
from tensorflow import set_random_seed
set_random_seed(3) # Tensorflow

from utils.utilities import Utilities
from causality_detection.feed_forward import FeedForward
from preprocessing.preprocesssor import Preprocessor


if __name__ == '__main__':
    utilities = Utilities()
    feed_forward = FeedForward()

    dataset_file = 'causal_pairs_dataset.csv'
    preprocessor = Preprocessor(['remove_stopwords', 'remove_non_letters', 'lemmatize'])

    data_rows = utilities.read_from_csv(dataset_file)
    del data_rows[0]
    X = []
    y = []

    for data_row in data_rows[:700]:
        candidate_causal_pair = eval(data_row[2])
        label = 1 if data_row[3] == 'causal' else 0

        candidate_causal_phrase = preprocessor.preprocess(candidate_causal_pair[0])
        candidate_effect_phrase = preprocessor.preprocess(candidate_causal_pair[1])
        if len(candidate_causal_phrase) > 0 and len(candidate_effect_phrase) > 0:
            X.append((candidate_causal_pair[0], candidate_causal_pair[1]))
            y.append(label)

    feed_forward.run(X, y)


