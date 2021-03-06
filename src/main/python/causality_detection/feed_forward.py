import math
import json
import tensorflow as tf
import networkx as nx
from scipy import interp
import matplotlib.pyplot as plt

from tensorflow import keras
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

from preprocessing.preprocesssor import Preprocessor
from preprocessing.event_detector import EventDetector
from utils.utilities import Utilities
from visualization.manage_results import ManageResults
from causality_detection.causal_stength_calculator import CausalStrengthCalculator
from causality_detection.causal_stength_calculator import MultiWordCausalStrengthCalculator

import numpy as np


class FeaturePreparation:
    def __init__(self, sentence_pairs):
        self.sentence_pairs = sentence_pairs
        self.preprocessor = Preprocessor(['remove_stopwords', 'remove_non_letters', 'lemmatize'])
        self.event_detector = EventDetector()

    def get_tokens(self):

        causal_sentences = [word_tokenize(self.preprocessor.preprocess(sentence_pair[0])) for sentence_pair in self.sentence_pairs]
        effect_sentences = [word_tokenize(self.preprocessor.preprocess(sentence_pair[1])) for sentence_pair in self.sentence_pairs]

        tokens_list = [causal_sentence + effect_sentence for causal_sentence, effect_sentence in zip(causal_sentences, effect_sentences)]
        return tokens_list

    def get_expanded_tokens(self, causal_net):
        all_expanded_tokens = []
        for causal_phrase, effect_phrase in self.sentence_pairs:
            causal_tokens = word_tokenize(self.preprocessor.preprocess(causal_phrase))
            expanded_tokens = []
            for token in causal_tokens:
                expanded_tokens.append(token)
                if causal_net.has_node(token):
                    successors = [successor for successor in causal_net.successors(token)]
                    if len(successors) > 0:
                        frequencies = []
                        for successor in successors:
                            frequencies.append(causal_net[token][successor]['freq'])

                        successors_with_frequencies = list(zip(successors, frequencies))
                        successors_with_frequencies.sort(key=lambda x: x[1], reverse=True)  # sort (desc) by freq values
                        successors, frequencies = zip(*tuple(successors_with_frequencies))
                        expanded_tokens += successors[:5]

            tokens = word_tokenize(self.preprocessor.preprocess(effect_phrase))

            for token in tokens:
                expanded_tokens.append(token)
                if causal_net.has_node(token):
                    predecessors = [predecessor for predecessor in causal_net.predecessors(token)]
                    if len(predecessors) > 0:
                        frequencies = []
                        for predecessor in predecessors:
                            frequencies.append(causal_net[predecessor][token]['freq'])

                        predecessors_with_frequencies = list(zip(predecessors, frequencies))
                        predecessors_with_frequencies.sort(key=lambda x: x[1], reverse=True)  # sort (desc) by freq values
                        predecessors, frequencies = zip(*tuple(predecessors_with_frequencies))
                        expanded_tokens += predecessors[:0]
            all_expanded_tokens.append(expanded_tokens)
        return all_expanded_tokens

    def get_expanded_event_tokens(self, causal_net, n_new_tokens):
        causal_phrases, effect_phrases = zip(*self.sentence_pairs)

        causal_events = self.event_detector.extract_event_from_sentences(causal_phrases)
        effect_events = self.event_detector.extract_event_from_sentences(effect_phrases)

        expanded_token_list = []
        for causal_event, effect_event in zip(causal_events, effect_events):
            causal_keyword = causal_event['keyword']
            effect_keyword = effect_event['keyword']

            causal_tokens = list(causal_event.values())
            effect_tokens = list(effect_event.values())

            if causal_net.has_node(effect_keyword):
                predecessors = [predecessor for predecessor in causal_net.predecessors(effect_keyword)]
                if len(predecessors) > 0:
                    frequencies = []
                    for predecessor in predecessors:
                        frequencies.append(causal_net[predecessor][effect_keyword]['freq'])

                    predecessors_with_frequencies = list(zip(predecessors, frequencies))
                    predecessors_with_frequencies.sort(key=lambda x: x[1], reverse=True)  # sort (desc) by freq values
                    predecessors, frequencies = zip(*tuple(predecessors_with_frequencies))
                    causal_tokens += predecessors[:n_new_tokens]

            if causal_net.has_node(causal_keyword):
                successors = [successor for successor in causal_net.successors(causal_keyword)]
                if len(successors) > 0:
                    frequencies = []
                    for successor in successors:
                        frequencies.append(causal_net[causal_keyword][successor]['freq'])

                    successors_with_frequencies = list(zip(successors, frequencies))
                    successors_with_frequencies.sort(key=lambda x: x[1], reverse=True)  # sort (desc) by freq values
                    successors, frequencies = zip(*tuple(successors_with_frequencies))
                    effect_tokens += successors[:n_new_tokens]
            expanded_token_list.append(causal_tokens + effect_tokens)

        return expanded_token_list

    def get_event_tokens(self):

        causal_phrases, effect_phrases = zip(*self.sentence_pairs)

        causal_events = self.event_detector.extract_event_from_sentences(causal_phrases)
        effect_events = self.event_detector.extract_event_from_sentences(effect_phrases)

        token_lists = []

        for causal_event, effect_event in zip(causal_events, effect_events):
            token_lists.append(list(causal_event.values()) + list(effect_event.values()))

        return token_lists

class Embedding:
    def __init__(self, n_dim, embedding_model_path):
        self.n_dim = n_dim
        self.embedding_model_path = embedding_model_path

    def get_embedding_matrix(self):
        model_path = self.embedding_model_path
        embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        vocabulary = [word for word in embedding_model.wv.vocab]

        embedding_index = {}
        embedding_matrix = []
        for index, word in enumerate(vocabulary):
            word_embedding = embedding_model.wv.get_vector(word)
            embedding_index[word] = word_embedding
            embedding_matrix.append(word_embedding)
        embedding_matrix = np.array(embedding_matrix)

        return vocabulary, embedding_matrix

    def get_index_vectors(self, vocabulary, tokens_list, maxlen):

        index_vectors = []
        for tokens in tokens_list:
            index_vector = []
            for token in tokens:
                try:
                    word_index = vocabulary.index(token)
                except ValueError:
                    word_index = vocabulary.index('UNK')
                index_vector.append(word_index)

            index_vectors.append(np.array(index_vector))

        index_vectors = keras.preprocessing.sequence.pad_sequences(np.array(index_vectors),
                                                                   value=vocabulary.index('UNK'),
                                                                   padding='post',
                                                                   maxlen=maxlen)
        return index_vectors


class HiddenLayer:
    def __init__(self, n_layer, vocab_size, embedding_matrix, dimension):
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.dimension = dimension

    def add_hidden_layer(self, model):
        layered_model = model
        layered_model.add(keras.layers.Embedding(self.vocab_size, self.dimension, weights=[self.embedding_matrix], trainable=True))
        layered_model.add(keras.layers.GlobalAveragePooling1D())
        layered_model.add(keras.layers.Dense(self.dimension, activation=tf.nn.relu))
        layered_model.add(keras.layers.Dense(200, activation=tf.nn.relu))
        layered_model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        layered_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        return layered_model


class CostFunction:
    def __init__(self, model):
        self.model = model

    def optimize(self, optimizer, learning_rate=0.1, loss='binary_crossentropy', metrics='accuracy'):
        optimized_model = self.model
        # setattr(optimizer, 'learning_rate', learning_rate)
        optimized_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

        return optimized_model


class Evaluation:
    def get_evaluation_data(self, dataset_file, n_pair):
        utilities = Utilities()

        preprocessor = Preprocessor(['remove_stopwords', 'remove_non_letters', 'lemmatize'])

        data_rows = utilities.read_from_csv(dataset_file)
        del data_rows[0]

        X = []
        y = []
        for data_row in data_rows[:n_pair]:
            candidate_causal_pair = eval(data_row[2])
            label = 1 if data_row[3] == 'causal' else 0

            candidate_causal_phrase = preprocessor.preprocess(candidate_causal_pair[0])
            candidate_effect_phrase = preprocessor.preprocess(candidate_causal_pair[1])
            if len(candidate_causal_phrase) > 0 and len(candidate_effect_phrase) > 0:
                X.append((candidate_causal_pair[0], candidate_causal_pair[1]))
                y.append(label)
        return X, y

    def run_experiment(self, settings):

        dataset_file = settings['dataset_file']
        result_file = settings['result_file']
        causal_net_file = settings['causal_net_file']
        embedding_model_file = settings['embedding_model_file']
        n_pair = settings['n_pair']
        n_expand = settings['n_expand']

        feed_forward = FeedForward()
        manage_results = ManageResults(result_file)

        feed_forward.causal_net_file = causal_net_file
        feed_forward.embedding_model_file = embedding_model_file

        print("Instances: %d, expand: %d" % (n_pair, n_expand))

        X, y = self.get_evaluation_data(dataset_file=dataset_file, n_pair=n_pair)

        preprocessed_dataset = {
            'X': X,
            'y': y
        }

        # save preprocessed causal pairs for evaluating Ponti's method
        with open('preprocessed_dataset.json', 'w') as outfile:
            json.dump(preprocessed_dataset, outfile)

        ff_result = feed_forward.run(X, y, n_expand)

        manage_results.save_dictionary_to_file(ff_result, settings['result_key'])

    def run_experiment_on_luos_method(self, settings):
        dataset_file = settings['dataset_file']
        result_file = settings['result_file']
        n_pair = settings['n_pair']
        threshold = settings['threshold']
        result_key = settings['result_key']

        manage_results = ManageResults(result_file)
        X, y = self.get_evaluation_data(dataset_file=dataset_file, n_pair=n_pair)

        causal_strength_calculator = CausalStrengthCalculator()
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f_score': []
        }

        for random_state in range(0, 1):
        # for random_state in range(0, 5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, stratify=y, random_state=random_state)
            y_pred = []
            for candidate in X_test:

                causal_score = causal_strength_calculator.get_causality_score(candidate[0], candidate[1])
                predicted_label = 1 if causal_score > threshold else 0
                y_pred.append(predicted_label)

            cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred))
            cv_scores['recall'].append(recall_score(y_test, y_pred))
            cv_scores['f_score'].append(f1_score(y_test, y_pred))

        result = {
            settings['result_key']: {
                'scores': ("%.2f" % (np.mean(cv_scores['accuracy']) * 100),
                           "%.2f" % (np.mean(cv_scores['precision']) * 100),
                           "%.2f" % (np.mean(cv_scores['recall']) * 100),
                           "%.2f" % (np.mean(cv_scores['f_score']) * 100)),

            }
        }

        print(result)

        manage_results.save_dictionary_to_file(result, result_key=result_key)

    def run_experiment_on_sasakis_method(self, settings):
        dataset_file = settings['dataset_file']
        result_file = settings['result_file']
        n_pair = settings['n_pair']
        threshold = settings['threshold']
        result_key = settings['result_key']

        manage_results = ManageResults(result_file)
        X, y = self.get_evaluation_data(dataset_file=dataset_file, n_pair=n_pair)

        causal_strength_calculator = MultiWordCausalStrengthCalculator()
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f_score': []
        }

        # for random_state in range(0, 5):
        for random_state in range(0, 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, stratify=y, random_state=random_state)
            y_pred = []
            for candidate in X_test:

                causal_score = causal_strength_calculator.get_causality_score(candidate[0], candidate[1])
                predicted_label = 1 if causal_score > threshold else 0
                y_pred.append(predicted_label)

            cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred))
            cv_scores['recall'].append(recall_score(y_test, y_pred))
            cv_scores['f_score'].append(f1_score(y_test, y_pred))

        result = {
            settings['result_key']: {
                'scores': ("%.2f" % (np.mean(cv_scores['accuracy']) * 100),
                           "%.2f" % (np.mean(cv_scores['precision']) * 100),
                           "%.2f" % (np.mean(cv_scores['recall']) * 100),
                           "%.2f" % (np.mean(cv_scores['f_score']) * 100)),

            }
        }

        print(result)

        manage_results.save_dictionary_to_file(result, result_key=result_key)

class Visualizer:
    def __init__(self, history):
        self.history = history
        self.acc = self.history.history['acc']
        self.val_acc = self.history.history['val_acc']
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.epochs = range(1, len(self.acc) + 1)

    def plot_loss_history(self):
        plt.clf()

        plt.plot(self.epochs, self.loss, 'r--', linewidth=1, label='Training loss')
        plt.plot(self.epochs, self.val_loss, 'g-', linewidth=1, label='Validation loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def plot_accuracy_history(self):
        plt.clf()   # clear figure

        plt.plot(self.epochs, self.acc, 'r--', linewidth=1, label='Training acc')
        plt.plot(self.epochs, self.val_acc, 'g-', linewidth=1, label='Validation acc')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


class FeedForward:
    def __init__(self):
        self.embedding_model_file = 'files/GoogleNews-vectors-negative300.bin'
        self.causal_net_file = 'causal_net_1m.pickle'

    def display_error_analysis(self, X_test, y_test, y_pred):
        false_positives = ['FP']
        false_negatives = ['FN']

        for index, x in enumerate(X_test):
            if y_test[index] == 1 and y_pred[index] == 0:
                false_negatives.append(x)
            elif y_test[index] == 0 and y_pred[index] == 1:
                false_positives.append(x)
            else:
                continue

        for fp in false_positives:
            print(fp)

        for fn in false_negatives:
            print(fn)

        return

    def run(self, X, y, n_expand):
        causal_net = nx.read_gpickle(self.causal_net_file)
        feature_preparation = FeaturePreparation(X)
        embedding = Embedding(n_dim=300, embedding_model_path=self.embedding_model_file)
        vocabulary, embedding_matrix = embedding.get_embedding_matrix()

        # tokens_list = feature_preparation.get_tokens()
        # tokens_list = feature_preparation.get_expanded_tokens(causal_net)
        # tokens_list = feature_preparation.get_event_tokens()
        tokens_list = feature_preparation.get_expanded_event_tokens(causal_net, n_expand)

        lengths = list(map(lambda x: len(x), tokens_list))
        padding_maxlen = math.ceil(sum(lengths)/len(lengths))
        index_vectors = embedding.get_index_vectors(vocabulary, tokens_list, maxlen=padding_maxlen)

        vocab_size = len(embedding_matrix)

        hidden_layer = HiddenLayer(n_layer=2, vocab_size=vocab_size, embedding_matrix=embedding_matrix, dimension=300)
        model = hidden_layer.add_hidden_layer(keras.Sequential())

        cost_function = CostFunction(model)

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
        model = cost_function.optimize(optimizer=optimizer)

        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f_score': []
        }

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        model_accuracy_training = []
        model_accuracy_validation = []
        model_loss_training = []
        model_loss_validation = []

        for random_state in range(0, 1):
        # for random_state in range(0, 5):
            X_train, X_test, y_train, y_test = train_test_split(index_vectors, y, test_size=0.40, stratify=y, random_state=random_state)

            history = model.fit(X_train, y_train, epochs=150, batch_size=40, validation_split=0.5, verbose=0)

            y_pred = [pred_class[0] for pred_class in model.predict_classes(X_test)]

            cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred))
            cv_scores['recall'].append(recall_score(y_test, y_pred))
            cv_scores['f_score'].append(f1_score(y_test, y_pred))

            # for ROC curve
            probas_ = model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_, pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            model_accuracy_training.append(history.history['acc'])
            model_accuracy_validation.append(history.history['val_acc'])
            model_loss_training.append(history.history['loss'])
            model_loss_validation.append(history.history['val_loss'])

            ## this code is only for error analysis. It has nothin to do with the actual evaluation
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=random_state)
            # self.display_error_analysis(X_test, y_test, y_pred)


        mean_tpr = np.mean(tprs, axis=0)

        mean_model_accuracy_training = [i*100 for i in np.mean(model_accuracy_training, axis=0)]
        mean_model_accuracy_validation = [i*100 for i in np.mean(model_accuracy_validation, axis=0)]
        mean_model_loss_training = [i*100 for i in np.mean(model_loss_training, axis=0)]
        mean_model_loss_validation = [i*100 for i in np.mean(model_loss_validation, axis=0)]

        result = {
            'scores': ("%.2f" % (np.mean(cv_scores['accuracy'])*100),
                       "%.2f" % (np.mean(cv_scores['precision'])*100),
                       "%.2f" % (np.mean(cv_scores['recall'])*100),
                       "%.2f" % (np.mean(cv_scores['f_score'])*100)),
            'roc': {
                'mean_tpr': mean_tpr.tolist(),
                'mean_fpr': mean_fpr.tolist()
            },
            'model_accuracy': {
                'training': mean_model_accuracy_training,
                'validation': mean_model_accuracy_validation
            },
            'model_loss': {
                'training': mean_model_loss_training,
                'validation': mean_model_loss_validation
            }

        }

        return result
