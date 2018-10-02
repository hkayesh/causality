from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from nltk.corpus import wordnet

from utils.utilities import Utilities
from causality_detection.causal_stength_calculator import CausalStrengthCalculator
from preprocessing.preprocesssor import Preprocessor


if __name__ == '__main__':
    causal_strength_calculator = CausalStrengthCalculator()

    utilities = Utilities()
    preprocessor = Preprocessor(['remove_stopwords', 'remove_non_letters', 'lemmatize'])
    dataset_file = 'causal_pairs_dataset.csv'

    data_rows = utilities.read_from_csv(dataset_file)
    del data_rows[0]
    X = []
    y_true = []
    y_pred = []
    threshold = 10

    for data_row in data_rows[:10]:
        candidate_causal_pair = eval(data_row[2])
        label = 1 if data_row[3] == 'causal' else 0

        candidate_causal_phrase = preprocessor.preprocess(candidate_causal_pair[0])
        candidate_effect_phrase = preprocessor.preprocess(candidate_causal_pair[1])
        if len(candidate_causal_phrase) > 0 and len(candidate_effect_phrase) > 0:
            causal_score = causal_strength_calculator.get_causality_score(candidate_causal_phrase, candidate_effect_phrase)
        else:
            causal_score = 0
        y_true.append(label)

        predicted_label = 1 if causal_score > threshold else 0
        y_pred.append(predicted_label)

        if label != predicted_label:
            print(candidate_causal_pair)
            print((label, predicted_label))

    print(precision_score(y_true, y_pred))
    print(recall_score(y_true, y_pred))
    print(f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


