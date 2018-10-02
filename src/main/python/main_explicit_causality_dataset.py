import re
from nltk.tokenize import sent_tokenize
from utils.utilities import Utilities
from preprocessing.event_extractor import EventExtractor
from preprocessing.event_detector import EventDetector
from preprocessing.preprocesssor import Preprocessor


def apply_causal_rules(sentence):

    sentence = sentence.lower()
    sentence_causal_pair = []

    phrase_expression = r'([\w:"\-/@#‘’\' ]+)'

    # Cues for the format: B cue_phrase A
    cues = ['caused by', 'result from', 'resulting from', 'results from', 'results from']
    cues += ['because of', ', because', 'because', ', inasmuch as', 'due to', 'in consequence of', 'owing to',
             'as a result of', 'as a consequence of']

    for cue in cues:
        cue = ' ' + cue + ' ' if cue[0] != ',' else cue + ' '
        matches = re.findall(phrase_expression + cue + phrase_expression, sentence)

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
    cues += [', thus', ', therefore', 'and hence', ', consequently', 'and consequently', ', for this reason alone,',
             ', hence']

    if len(sentence_causal_pair) < 1:
        for cue in cues:
            cue = ' ' + cue + ' ' if cue[0] != ',' else cue + ' '
            matches = re.findall(phrase_expression + cue + phrase_expression, sentence)

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

            matches = re.findall(cue_phrase_1 + phrase_expression + cue_phrase_2 + phrase_expression, sentence)
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

            matches = re.findall(cue_phrase_1 + phrase_expression + cue_phrase_2 + phrase_expression, sentence)
            if len(matches) > 0:
                reversed_matches = [(match[1], match[0]) for match in matches]
                sentence_causal_pair = reversed_matches
                break
    if len(sentence_causal_pair) > 0:
        output_causal_pair = sentence_causal_pair[0]
    else:
        output_causal_pair = None

    return output_causal_pair


def extract_sentence_event_pairs(text_causal_pair):
    event_pair = []
    cause_text = text_causal_pair[0]
    effect_text = text_causal_pair[1]

    # cause_text = 'Heavy rain'
    # effect_text = 'big Cyclone'

    causal_event = event_detector.extract_event_from_sentence(cause_text)
    effect_event = event_detector.extract_event_from_sentence(effect_text)

    if causal_event is not None and effect_event is not None:
        event_pair.append((causal_event, effect_event))

    return event_pair

def extract_batch_sentence_event_pairs(tweet_causal_rows):
    causal_phrases = []
    effect_phrases = []

    for tweet_causal_row in tweet_causal_rows:
        tweet_causal_pair = tweet_causal_row[2]
        causal_phrases.append(tweet_causal_pair[0])
        effect_phrases.append(tweet_causal_pair[1])

    causal_events = event_detector.extract_event_from_sentences(causal_phrases)
    effect_events = event_detector.extract_event_from_sentences(effect_phrases)

    print(causal_events)
    print(effect_events)

    print(len(causal_events))
    print(len(effect_events))

    counter = 0
    for causal_event, effect_event in zip(causal_events, effect_events):
        if causal_event is not None and effect_event is not None:
            counter +=1

    print(counter)

    return zip(causal_events, effect_events)


if __name__ == '__main__':

    utilities = Utilities()
    event_detector = EventDetector()
    event_extractor = EventExtractor()
    preprocessor = Preprocessor(['remove_urls', 'normalize'])

    tweet_rows = event_extractor.get_unique_tweets()

    header = tweet_rows[0]
    del tweet_rows[0]
    count = 0
    tweet_causal_rows = []
    for tweet_row in tweet_rows:

        tweet = preprocessor.preprocess(tweet_row[header.index('text')])
        sentences = sent_tokenize(tweet)
        # tweet_causal_pairs = []
        for sentence in sentences:
            sentence_causal_pair = apply_causal_rules(sentence)
            if sentence_causal_pair is not None:

                tweet_causal_rows.append([tweet_row, sentence, sentence_causal_pair])
                # print(sentence)
                # print(sentence_causal_pair)
                # if len(sentence_causal_pair)>1:
                #     print(sentence)
                #     print(sentence_causal_pair)
                #     print(len(sentence_causal_pair))
                #     count += 1

                # event_pair = extract_sentence_event_pairs(list(sentence_causal_pair))
                # if len(event_pair) > 0:
                #     event_pairs.append(event_pair)
                #     print(event_pair)
    # print(count)
    utilities.save_or_append_list_as_csv(tweet_causal_rows, 'causal_pairs.csv')
    # event_pairs = extract_batch_sentence_event_pairs(tweet_causal_rows)






