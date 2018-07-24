import os
import re
import json
# from sutime import SUTime

from utils.utilities import Utilities
from preprocessing.preprocesssor import Preprocessor
from required_files.config import app_config


class EventExtractor:
    def __init__(self):
        self.data_file = app_config['data_file']
        self.texts_in_file = 'texts_in_file.txt'
        self.ner_texts_file = 'output.txt'
        self.utilities = Utilities()
        self.preprocessor = Preprocessor(['remove_urls', 'remove_mentions', 'remove_hashtags', 'normalize'])

        jar_files = os.path.join(os.path.dirname(__file__), 'jars')
        # self.sutime = SUTime(jars=jar_files, mark_time_ranges=True)

    def save_texts_in_file(self):
        items = self.utilities.read_from_csv(self.data_file)

        header = items[0]

        texts = [item[header.index('text')] for item in items[1:]]

        processed_texts = [self.preprocessor.preprocess(text).encode('utf8') for text in texts]

        self.utilities.save_list_as_text_file(processed_texts, self.texts_in_file)

    def prepare_phrases(self, matches, tag, token_position=0, tag_position=-1, splitter='/'):
        phrases = []
        phrase = ''
        for match in matches:
            match_components = match.split(splitter)
            text_token = match_components[token_position].lower().strip()
            event_tag = match_components[tag_position]

            if event_tag == 'B-' + tag and len(phrase) < 1:
                phrase += text_token
            elif event_tag == 'B-' + tag and len(phrase) > 0:
                phrases.append(phrase)
                phrase = text_token
            else:
                phrase += ' ' + text_token
        phrases.append(phrase)
        phrases = list(set(phrases))

        return phrases

    def get_event_phrases(self, text):
        tag_name = 'EVENT'
        matches = re.findall(r'\w+/O/[A-Z]+/[BI]-' + tag_name, text)
        phrases = self.prepare_phrases(matches, tag_name)
        joined_text = ', '.join(phrases) if len(phrases) > 0 else ''

        return joined_text

    def get_event_locations(self, text):
        tag_name = 'geo-loc'
        matches = re.findall(r'\w+/[BI]-' + tag_name + '/[A-Z]+/O', text)
        phrases = self.prepare_phrases(matches=matches, tag=tag_name, token_position=0, tag_position=1)
        joined_text = ', '.join(phrases) if len(phrases) > 0 else ''

        return joined_text

    def get_event_entities(self, text):
        tag_names = ['person', 'company', 'facility', 'product', 'band', 'sportsteam', 'movie', 'tv-show']
        phrases = []
        for tag_name in tag_names:
            matches = re.findall(r'\w+/[BI]-' + tag_name + '/[A-Z]+/O', text)
            if len(matches) > 0:
                phrases += self.prepare_phrases(matches=matches, tag=tag_name, token_position=0, tag_position=1)

        joined_text = ', '.join(phrases) if len(phrases) > 0 else ''

        return joined_text

    def extract_events(self):
        data_rows = self.utilities.read_from_csv(self.data_file)
        text_rows = self.utilities.read_lines_from_file(self.ner_texts_file)

        header = data_rows[0]
        del data_rows[0]
        events = []
        for data_row, text_row in zip(data_rows, text_rows):

            event = {
                'tweet_id': data_row[header.index('id')],
                'entities': self.get_event_entities(text_row),
                'locations': self.get_event_locations(text_row),
                'event_time': data_row[header.index('created_at')],
                'event_phrases': self.get_event_phrases(text_row),
            }

            events.append(event)

        return events
