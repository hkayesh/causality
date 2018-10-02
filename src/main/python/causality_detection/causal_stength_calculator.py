import datetime
import networkx as nx
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

class CausalStrengthCalculator:
    def __init__(self):
        self.causal_net_path = 'causal_net_1m.pickle'
        self.causal_net = nx.read_gpickle(self.causal_net_path)
        self.N = len(self.causal_net.nodes())
        self.M = sum([edge[2]['freq'] for edge in self.causal_net.edges(data=True)])

    def get_prior_probas(self, i_c, j_e):
        prior_probas = {
            'p_of_i_c': 0,
            'p_of_j_e': 0,
            'p_of_i_c_and_j_e': 0
        }

        if self.causal_net.has_node(i_c) and self.causal_net.has_node(j_e) and self.causal_net.has_edge(i_c, j_e):
            f_of_i_and_j_e = self.causal_net[i_c][j_e]['freq']
            prior_probas['p_of_i_c_and_j_e'] = f_of_i_and_j_e/self.N

        if self.causal_net.has_node(i_c):
            number_of_pairs_with_i_c = sum([self.causal_net[i_c][successor]['freq'] for successor in self.causal_net.successors(i_c)])
            prior_probas['p_of_i_c'] = number_of_pairs_with_i_c/self.M

        if self.causal_net.has_node(j_e):
            number_of_pairs_with_j_e = sum([self.causal_net[predecessor][j_e]['freq'] for predecessor in self.causal_net.predecessors(j_e)])
            prior_probas['p_of_j_e'] = number_of_pairs_with_j_e/self.M

        return prior_probas

    def get_causal_strength(self, i_c, j_e, alpha=0.66, cs_lambda=0.5):
        cs_of_i_c_and_j_e = 0
        prior_probas = self.get_prior_probas(i_c, j_e)
        if prior_probas['p_of_i_c'] > 0 and prior_probas['p_of_j_e'] > 0:
            cs_nec_of_i_c_and_j_e = prior_probas['p_of_i_c_and_j_e'] / ((prior_probas['p_of_i_c'] ** alpha) * prior_probas['p_of_j_e'])
            cs_suf_of_i_c_and_j_e = prior_probas['p_of_i_c_and_j_e'] / (prior_probas['p_of_i_c'] * (prior_probas['p_of_j_e'] ** alpha))

            if cs_nec_of_i_c_and_j_e > 1 or cs_suf_of_i_c_and_j_e > 1:
                print('**')
                print(prior_probas)
                exit()
            cs_of_i_c_and_j_e = (cs_nec_of_i_c_and_j_e ** cs_lambda) * (cs_suf_of_i_c_and_j_e ** (1 - cs_lambda))

        return cs_of_i_c_and_j_e

    def get_causality_score(self, causal_candidate_phrase, effect_candidate_phrase):
        T_1 = word_tokenize(causal_candidate_phrase)
        T_2 = word_tokenize(effect_candidate_phrase)

        total_causal_strength = 0

        for i_c in T_1:
            for j_e in T_2:
                causal_strength = self.get_causal_strength(i_c, j_e)
                total_causal_strength += causal_strength

        causal_score = total_causal_strength / (len(T_1) + len(T_2))

        return causal_score

    def get_causality_score_from_synonyms(self, causal_candidate_phrase, effect_candidate_phrase):
        T_1_original = word_tokenize(causal_candidate_phrase)
        T_2_original = word_tokenize(effect_candidate_phrase)

        test_tokens = ['Web', 'server', 'designed', 'to', 'serve', 'HTTP', 'content', 'whereas', 'Application', 'server', 'is', 'not', 'limit', 'HTTP', 'response', 'but', 'also', 'can', 'serve', 'other', 'request']

        T_1 = T_1_original + test_tokens
        T_2 = T_2_original + test_tokens

        # T_1 = []
        # for token in T_1_original:
        #     synonyms = []
        #     for ss in wn.synsets(token):
        #         synonyms += ss.lemma_names()
        #     T_1 += list(set(synonyms))
        # if len(T_1) == 0:
        #     T_1 = T_1_original
        #
        # T_2 = []
        # for token in T_2_original:
        #     synonyms = []
        #     for ss in wn.synsets(token):
        #         synonyms += ss.lemma_names()
        #     T_2 += list(set(synonyms))
        # if len(T_2) == 0:
        #     T_2 = T_2_original

        total_causal_strength = 0

        for i_c in T_1:
            for j_e in T_2:
                causal_strength = self.get_causal_strength(i_c, j_e)
                total_causal_strength += causal_strength

        causal_score = total_causal_strength / (len(T_1) + len(T_2))

        return causal_score

    def get_causality_candidates(self, events_rows, header):

        chronological_event_rows = sorted(events_rows, key=lambda x: datetime.datetime.strptime(x[3], '%d-%m-%Y %H:%M'))

        causal_candidates = []
        for i, causal_event_row in enumerate(chronological_event_rows):
            text = causal_event_row[header.index('event_phrases')]
            terms = [term.strip() for term in text.split(',')]

            tokens = []
            for term in terms:
                tokens += word_tokenize(term)

            causal_terms = []
            for token in tokens:
                if self.causal_net.has_node(token):
                    causal_terms += [successor for successor in self.causal_net.successors(token)]

            for j, effect_event_row in enumerate(chronological_event_rows[i + 1:]):
                if causal_event_row[header.index('event_phrases')] != effect_event_row[header.index('event_phrases')]:
                    text = effect_event_row[header.index('event_phrases')]
                    tokens = word_tokenize(text)
                    for token in tokens:
                        if token in causal_terms:
                            causal_candidates.append((causal_event_row, effect_event_row))
                            break

        return causal_candidates

