import collections
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import word2vec
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from utils.utilities import Utilities
from preprocessing.event_extractor import EventExtractor

def tsne_plot_2d(model, label=False):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=0)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        if label is True:
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.show()

def tsne_plot_3d(model, label=False):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=0)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    z = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        z.append(value[2])

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i])
        if label is True:
            ax.text(x[i], y[i], z[i], labels[i])
    plt.show()

def hash(astring):
   return ord(astring[0])

def dataset_statistics(events):
    print('Total events: %d' % len(events))

    print('Total unique events: %d' % len(list(set(events))))

    frequent_events = collections.Counter(events).most_common(15)

    print('\nFrequent Events:')
    for frequent_event in frequent_events:
        print('%s\t%d' % frequent_event)


def perform_dbscan(X):
    db = DBSCAN(eps=0.10, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        print(class_member_mask)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':
    utilities = Utilities()
    event_extractor = EventExtractor()
    data_file = 'events.csv'
    data_rows = utilities.read_from_csv(data_file)

    events = []
    for data_row in data_rows:
        created_at = data_row[0]
        raw_event = eval(data_row[1])

        event_keys = raw_event.keys()
        if 'keyword' not in event_keys:
            continue
        keyword = raw_event['keyword']
        subj = raw_event['subj'] if 'subj' in event_keys else None
        dobj = raw_event['dobj'] if 'dobj' in event_keys else None
        prt = raw_event['prt'] if 'prt' in event_keys else None

        event_text = ''
        event_text = event_text + subj if subj is not None else event_text
        event_text = event_text + '-' + keyword if len(event_text) > 0 else keyword
        event_text = event_text + ' ' + prt if prt is not None else event_text
        event_text = event_text + '-' + dobj if dobj is not None else event_text

        events.append((created_at, raw_event, event_text))

    events_only = [event[2] for event in events]
    # dataset_statistics(events_only)


    # event_chunks = utilities.chunkify_list(events, 100)
    duration = 7*24*60*60
    event_chunks = event_extractor.chunkify_events_by_timeslots(events, duration)

    corpus = []

    for event_chunk in event_chunks:
        corpus_items = [chunk_item[2] for chunk_item in event_chunk]
        corpus.append(corpus_items)

    # corpus = [
    #     ['PERSON-have-holiday', 'PERSON-reach-LOCATION', 'family-go-camping', 'PERSON-watch-Television', 'PERSON-catch-BUS'],
    #     ['government-introduce-policy', 'PERSON-watch-Television', 'PERSON-criticize-policy', 'PERSON-watch-Television', 'people-criticize-government', 'PERSON-watch-Television', 'government-introduce-policy'],
    #     ['family-travel-abroad', 'PERSON-have-holiday', 'PERSON-watch-Television', 'PERSON-reach-LOCATION', 'PERSON-catch-BUS'],
    # ]

    model = word2vec.Word2Vec(corpus, size=300, window=5, min_count=10, workers=1, hashfxn=hash)

    cause_effect_pairs = []
    for candidate_causal_event in model.wv.vocab:
        candidate_effects = model.wv.most_similar(candidate_causal_event, topn=20)

        for candidate_effect_event, similarity_score in candidate_effects:
            # distance = model.wv.distance(candidate_effect_event, candidate_causal_event)
            candidate_causal_event_time = None
            candidate_effect_event_time = None
            number_of_precedence = 0

            for event_chunk in event_chunks:
                for event_item in event_chunk:
                    created_at = event_item[0]
                    event = event_item[2]
                    if event == candidate_causal_event:
                        candidate_causal_event_time = created_at
                    elif event == candidate_effect_event and candidate_causal_event_time is not None:
                        number_of_precedence += 1
                        break

            precedence_probability = number_of_precedence / len(event_chunks)
            causal_potential = (similarity_score + precedence_probability) / 2
            cause_effect_pairs.append((candidate_causal_event, candidate_effect_event, similarity_score, number_of_precedence, causal_potential))

    cause_effect_pairs.sort(key=lambda x: x[4], reverse=True)

    for sorted_cause_effect_pair in cause_effect_pairs[:100]:
        print(sorted_cause_effect_pair)





    exit()
    # tsne_plot_3d(model)

    # X = [model.wv.get_vector(event) for event in model.wv.vocab.keys()]
    # perform_dbscan(X[:50000])