import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

from tensorflow import keras
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

from preprocessing.preprocesssor import Preprocessor
from preprocessing.event_detector import EventDetector

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

    # def get_expanded_tokens(self, causal_net):
    #     causal_phrases, effect_phrases = zip(*self.sentence_pairs)
    #     causal_events = self.event_detector.extract_event_from_sentences(causal_phrases)
    #
    #     print(causal_events)
    #     exit()
    #
    #     expanded_tokens = []
    #     for token in input_tokens:
    #         expanded_tokens.append(token)
    #         if causal_net.has_node(token):
    #             successors = [successor for successor in causal_net.successors(token)]
    #             predecessors = [predecessor for predecessor in causal_net.predecessors(token)]
    #             expanded_tokens += successors
    #             expanded_tokens += predecessors
    #
    #     return expanded_tokens


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

    def get_index_vectors(self, vocabulary, tokens_list):

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
                                                                   maxlen=10)
        return index_vectors


class HiddenLayer:
    def __init__(self, n_layer, vocab_size, embedding_matrix, dimension):
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.dimension = dimension

    def add_hidden_layer(self, model):
        layered_model = model
        layered_model.add(keras.layers.Embedding(self.vocab_size, self.dimension, weights=[self.embedding_matrix]))
        layered_model.add(keras.layers.GlobalAveragePooling1D())
        layered_model.add(keras.layers.Dense(self.dimension, activation=tf.nn.tanh))
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
    def __init__(self, score_type):
        self.score_type = score_type


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

        # "bo" is for "blue dot"
        plt.plot(self.epochs, self.loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(self.epochs, self.val_loss, 'b', label='Validation loss')
        # plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def plot_accuracy_history(self):
        plt.clf()   # clear figure

        plt.plot(self.epochs, self.acc, 'bo', label='Training acc')
        plt.plot(self.epochs, self.val_acc, 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


class FeedForward:

    # def get_embedding_dict(self):
    #     model_path = 'files/GoogleNews-vectors-negative300.bin'
    #     embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    #
    #     embedding_index = {}
    #     vocabulary = embedding_model.wv.vocab
    #     embedding_matrix = np.zeros(len(vocabulary), 300)
    #     for index, word in enumerate(vocabulary):
    #         word_embedding = embedding_model.wv.get_vector(word)
    #         embedding_index[word] = word_embedding
    #         embedding_matrix[index] = word_embedding
    #
    #     return embedding_matrix
    #
    # def convert_to_vector(self, sentences):
    #     #  downloaded from http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    #     model_path = 'files/GoogleNews-vectors-negative300.bin'
    #     embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    #     unknown_word = 'UNK'
    #
    #     sentence_embeddings = []
    #     for sentence in sentences:
    #         word_embeddings = []
    #         for word in sentence:
    #             try:
    #                 embedding = embedding_model.get_vector(word)
    #                 word_embeddings.append(list(embedding))
    #             except KeyError:
    #                 word_embeddings.append(embedding_model.get_vector(unknown_word))
    #
    #         sentence_embeddings.append(word_embeddings)
    #
    #     return sentence_embeddings
    #
    # def get_features(self, sentence_pairs):
    #     causal_sentences = [word_tokenize(sentence_pair[0]) for sentence_pair in sentence_pairs]
    #     effect_sentences = [word_tokenize(sentence_pair[1]) for sentence_pair in sentence_pairs]
    #
    #     causal_sentences_vectors = self.convert_to_vector(causal_sentences)
    #     effect_sentences_vectors = self.convert_to_vector(effect_sentences)
    #
    #     combined_vectors = []
    #     for causal_sentence_vectors, effect_sentence_vectors in zip(causal_sentences_vectors, effect_sentences_vectors):
    #         combined_vectors.append(causal_sentence_vectors + effect_sentence_vectors)
    #
    #     return combined_vectors

    def run(self, X, y):
        causal_net_path = 'causal_net_1m.pickle'
        causal_net = nx.read_gpickle(causal_net_path)
        feature_preparation = FeaturePreparation(X)
        embedding = Embedding(n_dim=300, embedding_model_path='files/GoogleNews-vectors-negative300.bin')
        vocabulary, embedding_matrix = embedding.get_embedding_matrix()

        tokens_list = feature_preparation.get_tokens()
        # tokens_list = feature_preparation.get_expanded_tokens(causal_net)

        index_vectors = embedding.get_index_vectors(vocabulary, tokens_list)

        X_train, X_test, y_train, y_test = train_test_split(index_vectors, y, test_size=0.40, random_state=1)

        vocab_size = len(embedding_matrix)

        hidden_layer = HiddenLayer(n_layer=2, vocab_size=vocab_size, embedding_matrix=embedding_matrix, dimension=300)
        model = hidden_layer.add_hidden_layer(keras.Sequential())

        cost_function = CostFunction(model)

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
        model = cost_function.optimize(optimizer=optimizer)

        history = model.fit(X_train,
                            y_train,
                            epochs=40,
                            batch_size=10,
                            validation_split=0.5,
                            verbose=1)

        results = model.evaluate(X_test, y_test)

        print(results)

        visualizer = Visualizer(history=history)
        visualizer.plot_loss_history()
        visualizer.plot_accuracy_history()
