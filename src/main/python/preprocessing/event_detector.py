import itertools
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tree import ParentedTree

from preprocessing.preprocesssor import Preprocessor
from utils.utilities import Utilities


class EventDetector:
    def __init__(self):
        self.path_to_jar = 'lib/stanford_parser/stanford-parser.jar'
        self.path_to_models_jar = 'lib/stanford_parser/stanford-english-corenlp-2018-02-27-models.jar'
        self.path_to_ner_tagger = 'lib/stanford_ner/stanford-ner.jar'
        self.path_to_ner_model = 'lib/stanford_ner/english.all.3class.distsim.crf.ser.gz'

        self.ner_tagger = StanfordNERTagger(self.path_to_ner_model, self.path_to_ner_tagger)
        self.dependency_parser = StanfordDependencyParser(path_to_jar=self.path_to_jar, path_to_models_jar=self.path_to_models_jar)
        self.lemmatizer = WordNetLemmatizer()
        self.utilities = Utilities()

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

        event = None
        for verb in list(raw_events.keys()):
            event_info = raw_events[verb]
            if len(verb) < 2 or 'subj' not in list(event_info.keys()) or len(event_info['subj']) < 2 \
                    or 'dobj' not in list(event_info.keys()) or len(event_info['dobj']) < 2:
                continue

            event_info['keyword'] = verb
            event = event_info
            break  # return only one event

        return event

    def extract_soft_events(self, dependency_tree, dependency_relations, ner_tags):

        entity_categories = ['PERSON', 'LOCATION', 'ORGANIZATION']
        accepted_relation_keys = ['nsubj', 'nsubjpass', 'amod', 'dobj', 'advmod', 'nmod', 'xcomp', 'compound:prt', 'compound', 'neg']

        keyword = self.lemmatizer.lemmatize(dependency_tree.label(), 'v')

        event = {'keyword': keyword}
        for dependency_relation in dependency_relations:
            if len(dependency_relation) == 3:
                head = dependency_relation[0]
                relation = dependency_relation[1]
                tail = dependency_relation[2]

                if head[0] == keyword and relation in accepted_relation_keys:
                    event[relation] = self.lemmatizer.lemmatize(tail[0].lower())
        # print(event)
        return event

    def extract_event_from_sentence(self, sentence):
        event = None
        sentence_preprocessor = Preprocessor(['remove_non_letters'])

        processed_sentence = sentence_preprocessor.preprocess(sentence)

        sent_dependencies = self.dependency_parser.raw_parse(processed_sentence)
        sent_ner_tags = self.ner_tagger.tag_sents([processed_sentence.split()])
        dependencies = [list(parse.triples()) for parse in sent_dependencies]

        if len(dependencies) > 0 and dependencies[0] is not None:
            event = self.extract_events_from_stanford_dependencies(dependencies[0], sent_ner_tags)
        else:
            event['keyword'] = sentence

        return event

    def extract_event_from_sentences(self, sentences):
        events = []
        sentence_preprocessor = Preprocessor(['remove_non_letters'])

        chunks = list(self.utilities.chunkify_list(data_list=sentences, items_per_chunk=1000))

        for chunk in chunks:
            sentences = []
            for chunk_item in chunk:
                sentences.append(sentence_preprocessor.preprocess(chunk_item))

            chunk_sent_dependencies = self.dependency_parser.raw_parse_sents(sentences)
            chunk_sent_ner_tags = self.ner_tagger.tag_sents([sentence.split() for sentence in sentences])

            for sent_dependencies, sent_ner_tags, sentence in zip(chunk_sent_dependencies, chunk_sent_ner_tags, sentences):
                temp_sent_dependencies_1, temp_sent_dependencies_2 = itertools.tee(sent_dependencies, 2)
                dependency_relations = [list(parse.triples()) for parse in temp_sent_dependencies_1]
                dependency_tree = [parse.tree() for parse in temp_sent_dependencies_2][0]

                if len(dependency_relations) > 0 and dependency_relations[0] is not None and len(dependency_relations[0]) > 0:
                    # print(sentence)
                    event = self.extract_soft_events(dependency_tree, dependency_relations[0], sent_ner_tags)
                else:
                    event = {'keyword': sentence}

                events.append(event)

        return events


