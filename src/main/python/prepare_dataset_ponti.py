import sys
import json

from nltk.tokenize import word_tokenize
from preprocessing.preprocesssor import Preprocessor
from nltk.stem.wordnet import WordNetLemmatizer
from utils.utilities import Utilities


from preprocessing.event_detector import EventDetector

if __name__ == '__main__':
    event_detector = EventDetector()
    preprocessor = Preprocessor(['remove_non_letters', 'lemmatize'])
    lemmatizer = WordNetLemmatizer()
    utilities = Utilities()

    input_dataset_file = 'preprocessed_dataset.json'
    output_dataset_file = 'preprocessed_dataset_ponti.txt'

    with open(input_dataset_file) as json_file:
        data = json.load(json_file)

    X = data['X']
    y = data['y']

    causal_phrases, effect_phrases = zip(*X)

    causal_events = event_detector.extract_event_from_sentences(causal_phrases)
    effect_events = event_detector.extract_event_from_sentences(effect_phrases)

    items = []
    for causal_phrase, effect_phrase, causal_event, effect_event, label in zip(causal_phrases, effect_phrases, causal_events, effect_events, y):
        causal_phrase = preprocessor.preprocess(causal_phrase)
        effect_phrase = preprocessor.preprocess(effect_phrase)
        causal_event_attributes = []

        for key in causal_event.keys():
            if key is not 'keyword':
                causal_event_attributes.append(causal_event[key])

        effect_event_attributes = []
        for key in effect_event.keys():
            if key is not 'keyword':
                effect_event_attributes.append(effect_event[key])

        causal_tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(token, 'v')) for token in word_tokenize(causal_phrase)]
        effect_tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(token, 'v')) for token in word_tokenize(effect_phrase)]
        causal_keyword = causal_event['keyword']
        effect_keyword = effect_event['keyword']

        try:
            causal_keyword_index = causal_tokens.index(lemmatizer.lemmatize(causal_keyword))
        except ValueError:
            causal_keyword_index = 0
        try:
            effect_keyword_index = effect_tokens.index(lemmatizer.lemmatize(effect_keyword))
        except ValueError:
            effect_keyword_index = 0

        total_tokens = len(causal_tokens) + len(effect_tokens)

        left_distances = []
        right_distances = []
        for i, j in zip(range(-1*causal_keyword_index, total_tokens), range(-1*(effect_keyword_index + len(causal_tokens)), total_tokens)):
            distance = str(abs(i)) + '-' + str(abs(j))
            if len(left_distances) < len(causal_tokens):
                left_distances.append(distance)
            elif len(right_distances) < len(effect_tokens):
                right_distances.append(distance)
            else:
                break

        item = [causal_phrase, effect_phrase, causal_keyword, effect_keyword,
                ' '.join(causal_event_attributes), ' '.join(effect_event_attributes), '|'.join(left_distances),
                '|'.join(right_distances), str(label)]
        items.append('\t'.join(item))

    utilities.save_list_as_text_file(items, output_dataset_file)






