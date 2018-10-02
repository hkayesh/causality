import os
import re
import json
import time
from datetime import datetime, timedelta
# from sutime import SUTime
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser

from utils.utilities import Utilities
from preprocessing.preprocesssor import Preprocessor
from required_files.config import app_config


class EventExtractor:
    def __init__(self):
        self.data_file = app_config['data_file']
        self.texts_in_file = 'texts_in_file.txt'
        self.ner_texts_file = 'output.txt'
        self.utilities = Utilities()
        self.lemmatizer = WordNetLemmatizer()
        self.preprocessor = Preprocessor(['remove_urls', 'remove_mentions', 'remove_hashtags', 'normalize'])

        # jar_files = os.path.join(os.path.dirname(__file__), 'jars')
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
        unique_texts = []
        for data_row, text_row in zip(data_rows, text_rows):
            text = self.preprocessor.preprocess(data_row[header.index('text')])

            if text in unique_texts:
                continue

            event = {
                'tweet_id': data_row[header.index('id')],
                'entities': self.get_event_entities(text_row),
                'locations': self.get_event_locations(text_row),
                'event_time': data_row[header.index('created_at')],
                'event_phrases': self.get_event_phrases(text_row),
            }

            events.append(event)
            unique_texts.append(text)

        return events

    def extract_events_from_stanford_dependencies(self, dependencies, ner_tags):
        entity_categories = ['PERSON', 'LOCATION', 'ORGANIZATION']
        raw_events = {}
        for dependency in dependencies:
            if len(dependency) == 3:
                head = dependency[0]
                relation = dependency[1]
                tail = dependency[2]

                if head[1].startswith('VB'):
                    event_keywords = list(raw_events.keys())
                    event_keyword = self.lemmatizer.lemmatize(head[0].lower(), 'v')
                    if event_keyword not in event_keywords:
                        raw_events[event_keyword] = {}

                    if relation.endswith('subj'):
                        subject_pronoun = ['i', 'you', 'he', 'she', 'we', 'they', 'who']
                        subj_value = self.lemmatizer.lemmatize(tail[0].lower())

                        if tail[0].lower() in subject_pronoun:
                            subj_value = 'PERSON'
                        else:
                            for ner_tag in ner_tags:
                                if ner_tag[0] == tail[0] and ner_tag[1] in entity_categories:
                                    subj_value = ner_tag[1]
                        raw_events[event_keyword]['subj'] = subj_value

                    if relation == 'dobj':
                        objective_pronoun = ['me', 'you', 'him', 'her', 'us', 'you', 'them']
                        dobj_value = self.lemmatizer.lemmatize(tail[0].lower())

                        if tail[0].lower() in objective_pronoun:
                            dobj_value = 'PERSON'
                        else:
                            for ner_tag in ner_tags:
                                if ner_tag[0] == tail[0] and ner_tag[1] in entity_categories:
                                    dobj_value = ner_tag[1]

                        raw_events[event_keyword]['dobj'] = dobj_value

                    if relation == 'compound:prt':
                        raw_events[event_keyword]['prt'] = tail[0]

        events = []
        for verb in list(raw_events.keys()):
            event = raw_events[verb]
            if len(verb) < 2 or 'subj' not in list(event.keys()) or len(event['subj']) < 2 \
                    or 'dobj' not in list(event.keys()) or len(event['dobj']) < 2:
                continue

            event['keyword'] = verb
            events.append(event)

        return events

    def get_unique_tweets(self, n_rows=None):
        data_rows = self.utilities.read_from_csv(self.data_file)
        preprocessor = Preprocessor(['remove_urls', 'remove_mentions', 'remove_hashtags', 'normalize', 'remove_non_letters'])

        header = data_rows[0]
        del data_rows[0]
        tweet_rows = {}
        for data_row in data_rows:
            if n_rows is not None and len(tweet_rows) >= n_rows:
                break
            tweet = preprocessor.preprocess(data_row[header.index('text')])
            if tweet not in list(tweet_rows.keys()):
                tweet_rows[tweet] = data_row
        tweet_rows = [header] + list(tweet_rows.values())

        return tweet_rows

    def get_tweet_sentences(self, tweet_rows):
        header = tweet_rows[0]
        del tweet_rows[0]

        tweet_sentences = []
        for tweet_row in tweet_rows:
            created_at = tweet_row[header.index('created_at')]
            text = self.preprocessor.preprocess(tweet_row[header.index('text')])
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if len(sentence) > 1:
                    tweet_sentences.append((created_at, sentence))

        return tweet_sentences

    def extract_events2(self, tweet_sentences):
        path_to_jar = 'lib/stanford_parser/stanford-parser.jar'
        path_to_models_jar = 'lib/stanford_parser/stanford-english-corenlp-2018-02-27-models.jar'
        path_to_ner_tagger = 'lib/stanford_ner/stanford-ner.jar'
        path_to_ner_model = 'lib/stanford_ner/english.all.3class.distsim.crf.ser.gz'

        sentence_preprocessor = Preprocessor(['remove_non_letters'])
        ner_tagger = StanfordNERTagger(path_to_ner_model, path_to_ner_tagger)
        dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

        events = []

        chunks = list(self.utilities.chunkify_list(data_list=tweet_sentences, items_per_chunk=1000))

        for chunk in chunks:
            created_ats = []
            sentences = []
            for chunk_item in chunk:
                created_ats.append(chunk_item[0])
                sentences.append(sentence_preprocessor.preprocess(chunk_item[1]))

            chunk_sent_dependencies = dependency_parser.raw_parse_sents(sentences)
            chunk_sent_ner_tags = ner_tagger.tag_sents([sentence.split() for sentence in sentences])

            for sent_dependencies, sent_ner_tags, created_at in zip(chunk_sent_dependencies, chunk_sent_ner_tags, created_ats):
                dependencies = [list(parse.triples()) for parse in sent_dependencies]

                if len(dependencies) > 0 and dependencies[0] is not None:
                    sentence_events = self.extract_events_from_stanford_dependencies(dependencies[0], sent_ner_tags)
                    if len(sentence_events) > 0:
                        for sentence_event in sentence_events:
                            events.append((created_at, sentence_event))

        return events

    def chunkify_events_by_timeslots(self, events, duration):
        slot_starts_at = None
        event_chunks = []
        event_chunk = []
        for event in events:
            created_at = datetime.strptime(event[0], '%d-%m-%Y %H:%M')

            if slot_starts_at is None:
                slot_starts_at = created_at

            if len(event_chunk) > 0 and created_at > slot_starts_at + timedelta(0, duration):
                event_chunks.append(event_chunk)
                event_chunk = []
                slot_starts_at = created_at
            event_chunk.append(event)
        event_chunks.append(event_chunk)
        return event_chunks
