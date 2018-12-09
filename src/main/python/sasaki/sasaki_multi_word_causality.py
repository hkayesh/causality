import csv


class SasakiMultiWordCausality(object):
    def __init__(self):
        self.wiktionary_file = 'sasaki/wikitionary_20121127.tsv'  # https://semisignal.com/wiktionary-definitions-database/


    def get_multi_word_verbs(self):
        multi_word_verbs = []
        with open(self.wiktionary_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                word = row[1]
                pos = row[2]
                if pos == 'Verb':
                    verb_splits = word.split(' ')
                    if 2 <= len(verb_splits) <= 3:
                        multi_word_verb = ' '.join(verb_splits)
                        multi_word_verbs.append(multi_word_verb)
        return list(set(multi_word_verbs))




