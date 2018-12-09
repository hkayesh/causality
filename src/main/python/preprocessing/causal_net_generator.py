import re
import csv
import json
import networkx as nx
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from smart_open import smart_open

from utils.utilities import Utilities
from preprocessing.preprocesssor import Preprocessor
from sasaki.sasaki_multi_word_causality import SasakiMultiWordCausality


class CausalNetGenerator:

    def __init__(self):
        self.wiki_file = '/home/humayun/enwiki-latest-pages-articles.xml.bz2'
        self.utilities = Utilities()
        self.preprocessor = Preprocessor(['remove_stopwords', 'remove_punctuation', 'lemmatize'])

    def get_articles(self, number=None, offset=0):
        full_texts = []
        count = 0
        for index, line in enumerate(smart_open('enwiki-10072018.json.gz')):
            if index < offset:
                continue
            if number is not None and count >= number:
                break

            article = json.loads(line)

            full_text = ''
            for section_title, section_text in zip(article['section_titles'], article['section_texts']):

                full_text = full_text + ' ' + section_text.strip()

            full_texts.append(full_text)
            count += 1

        return full_texts

    def apply_causal_rules(self, text):
        sentences = sent_tokenize(text)

        causal_pairs = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence_causal_pair = []

            # Cues for the format: B cue_phrase A
            cues = ['caused by', 'result from', 'resulting from', 'results from', 'results from']
            cues += ['because of', ', because',  'because', ', inasmuch as', 'due to', 'in consequence of', 'owing to',
                     'as a result of', 'as a consequence of']

            for cue in cues:
                cue = ' ' + cue + ' ' if cue[0] != ',' else cue + ' '
                matches = re.findall(r'([\w ]+)' + cue + r'([\w ]+)', sentence)

                if len(matches) > 0:
                    reversed_matches = [(match[1], match[0]) for match in matches]
                    sentence_causal_pair = reversed_matches
                    break

            # Cues for the format: A cue_phrase B
            cues = ['lead to', 'leads to', 'led to',
                    'leading to', 'give rise to', 'gave rise to',
                    'given rise to', 'giving rise to', 'induce',
                    'inducing', 'induces', 'induced',
                    'cause', 'causing', 'causes',
                    'caused', 'bring on', 'brought on',
                    'bringing on', 'brings on']
            cues += [', thus', ', therefore', 'and hence', ', consequently', 'and consequently', ', for this reason alone,', ', hence']

            if len(sentence_causal_pair) < 1:
                for cue in cues:
                    cue = ' ' + cue + ' ' if cue[0] != ',' else cue + ' '
                    matches = re.findall(r'([\w ]+)' + cue + r'([\w ]+)', sentence)

                    if len(matches) > 0:
                        sentence_causal_pair = matches
                        break

            # # Cues for the format: cue_phrase1 A cue_phrase2 B
            cues = [('if', ', then'), ('if', ','), ('in consequence of', ','), ('owing to', ',')]
            cues += [('the effect of', 'is'), ('the effect of', 'was'), ('the effect of', 'will')]

            if len(sentence_causal_pair) < 1:
                for cue in cues:
                    cue_phrase_1 = cue[0] + ' '
                    cue_phrase_2 = cue[1] + ' '

                    matches = re.findall(cue_phrase_1 + '([\w ]+)' + cue_phrase_2 + '([\w ]+)', sentence)
                    if len(matches) > 0:
                        sentence_causal_pair = matches
                        break

            # cues for the format: the reason(s) of/for B, A
            cues = [('the reason of', 'is'), ('the reason of', 'was'), ('the reasons of', 'are'),
                    ('the reasons of', 'were'), ('the reason for', 'is'), ('the reason for', 'was'),
                    ('the reasons for', 'are'), ('the reasons for', 'were')]

            if len(sentence_causal_pair) < 1:
                for cue in cues:
                    cue_phrase_1 = cue[0] + ' '
                    cue_phrase_2 = ' ' + cue[1] + ' '

                    matches = re.findall(cue_phrase_1 + '([\w ]+)' + cue_phrase_2 + '([\w ]+)', sentence)
                    if len(matches) > 0:
                        reversed_matches = [(match[1], match[0]) for match in matches]
                        sentence_causal_pair = reversed_matches
                        break

            causal_pairs += sentence_causal_pair

        return causal_pairs

    def remove_unnecessary_words(self, causal_pairs):

        accepted_pos_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS', 'ADV']
        clean_causal_pairs = []
        for causal_pair in causal_pairs:
            cause_phrase = causal_pair[0]
            effect_phrase = causal_pair[1]
            cause_tokens = word_tokenize(cause_phrase)
            pos_tagged_tokens = pos_tag(cause_tokens)
            cause_clean_tokens = []
            effect_clean_tokens = []

            for token_tag in pos_tagged_tokens:
                if token_tag[1] in accepted_pos_tags:
                    cause_clean_tokens.append(token_tag[0])

            effect_tokens = word_tokenize(effect_phrase)
            pos_tagged_tokens = pos_tag(effect_tokens)

            for token_tag in pos_tagged_tokens:
                if token_tag[1] in accepted_pos_tags:
                    effect_clean_tokens.append(token_tag[0])

            cause_clean_tokens = self.preprocessor.preprocess(' '.join(cause_clean_tokens))
            effect_clean_tokens = self.preprocessor.preprocess(' '.join(effect_clean_tokens))

            if len(cause_clean_tokens) > 0 and len(effect_clean_tokens) > 0:
                clean_causal_pairs.append((cause_clean_tokens, effect_clean_tokens))

        return clean_causal_pairs

    def get_causal_pair_phrases(self, article):
        article_causal_pairs = self.apply_causal_rules(article)
        article_causal_pairs = self.remove_unnecessary_words(article_causal_pairs)

        return article_causal_pairs

    def get_causal_pair_tokens(self, causal_pair_phrase):
        cause_tokens = word_tokenize(causal_pair_phrase[0])
        effect_tokens = word_tokenize(causal_pair_phrase[1])
        causal_pairs = []
        for cause_token in cause_tokens:
            cause_replicated_list = [cause_token] * len(effect_tokens)

            causal_pairs += list(zip(cause_replicated_list, effect_tokens))

        return causal_pairs

    def create_or_update_directed_causal_graph(self, causal_pairs, graph=None):
        if graph is None:
            graph = nx.DiGraph()

        for causal_pair in causal_pairs:
            cause_token = causal_pair[0]
            effect_token = causal_pair[1]

            if cause_token not in graph:
                graph.add_node(cause_token)
            if effect_token not in graph:
                graph.add_node(effect_token)

            if graph.has_successor(cause_token, effect_token):
                graph[cause_token][effect_token]['freq'] += 1
            else:
                graph.add_edge(cause_token, effect_token)
                graph[cause_token][effect_token]['freq'] = 1

        return graph

    def get_all_causal_pair_tokens(self, articles):
        causal_pair_tokens = []

        for article in articles:
            causal_pair_phrases = self.get_causal_pair_phrases(article)

            # utilities.save_or_append_list_as_csv(causal_pair_phrases, 'causal_pair_phrases.csv')

            for causal_pair_phrase in causal_pair_phrases:
                causal_pair_tokens += self.get_causal_pair_tokens(causal_pair_phrase)

            # utilities.save_or_append_list_as_csv(causal_pair_tokens, 'causal_pair_tokens.csv')
        return causal_pair_tokens


class CausalNetGeneratorFromNews(CausalNetGenerator):
    def __init__(self):
        super(CausalNetGenerator, self).__init__()
        self.article_dump_file = 'files/signalmedia-1m.jsonl'

    def get_articles(self, number=None, offset=0):
        full_texts = []
        count = 0
        for index, line in enumerate(smart_open(self.article_dump_file)):
            if index < offset:
                continue
            if number is not None and count >= number:
                break
            article = json.loads(line)

            full_texts.append(article['content'].strip())
            count += 1

        return full_texts


class MultiWordCausalNetGeneratorFromNews(CausalNetGeneratorFromNews):
    def __init__(self):
        super(CausalNetGeneratorFromNews, self).__init__()
        self.article_dump_file = 'files/signalmedia-1m.jsonl'  # https://research.signalmedia.co/newsir16/signal-dataset.html
        self.sasaki_multi_word_causality = SasakiMultiWordCausality()
        self.multi_word_verbs = self.sasaki_multi_word_causality.get_multi_word_verbs()

    def get_causal_pair_tokens(self, causal_pair_phrase):
        cause_tokens = word_tokenize(causal_pair_phrase[0])
        effect_tokens = word_tokenize(causal_pair_phrase[1])
        for verb in self.multi_word_verbs:
            if re.search(r'\b' + verb + r'\b', causal_pair_phrase[0]):
                cause_tokens.append(verb)

            if re.search(r'\b' + verb + r'\b', causal_pair_phrase[1]):
                effect_tokens.append(verb)

        causal_pairs = []
        for cause_token in cause_tokens:
            cause_replicated_list = [cause_token] * len(effect_tokens)
            causal_pairs += list(zip(cause_replicated_list, effect_tokens))

        return causal_pairs
