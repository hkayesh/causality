import timeit
import collections

from utils.utilities import Utilities
from preprocessing.preprocesssor import Preprocessor
from causality_detection.causal_stength_calculator import CausalStrengthCalculator
from causality_detection.itemsest_causality import ItemsetCausality


if __name__ == "__main__":
    start_time = timeit.default_timer()
    event_file_path = 'events.csv'
    utilities = Utilities()
    causal_strength_calculator = CausalStrengthCalculator()
    itemset_causality = ItemsetCausality()
    preprocessor = Preprocessor(params=['lower', 'lemmatize'])

    rows = utilities.read_from_csv(event_file_path)
    header = rows[0]
    del rows[0]

    events_phrases = []
    for row in rows:
        phrases = [phrase.strip() for phrase in row[header.index('event_phrases')].split(',')]
        events_phrases += phrases

    sorted_event_phrases = collections.Counter(events_phrases).most_common()
    low_freq_events = [event[0] for event in sorted_event_phrases if event[1] <= 5]

    event_rows = []
    for row in rows:
        event_tokens = row[header.index('event_phrases')].split(',')
        only_frequent_event_tokens = [event_token for event_token in event_tokens if event_token not in low_freq_events]
        if len(only_frequent_event_tokens) > 0:
            row[header.index('event_phrases')] = ','.join(only_frequent_event_tokens)
            event_rows.append(row)

    meta_info = {}
    for row in event_rows[:100]:
        event_tokens = row[header.index('event_phrases')].split(',')
        meta_info[len(event_tokens)] = 1 if len(event_tokens) not in list(meta_info.keys()) else meta_info[len(event_tokens)] + 1

        # for event_token in event_tokens:
        #     meta_info[event_token] = 1 if event_token not in list(meta_info.keys()) else meta_info[event_token] + 1

        # if len(row[header.index('entities')]) > 0 and row[header.index('locations')] and len(event_tokens) == 1:
        #    event_rows.append(row)
    filtered_events = []
    for event_row in event_rows[:100]:
        event_tokens = event_row[header.index('event_phrases')].split(',')
        if len(event_row[header.index('entities')]) > 0 and event_row[header.index('locations')] and len(event_tokens) == 1:
            event_row[header.index('event_phrases')] = preprocessor.preprocess(event_row[header.index('event_phrases')].strip())
            filtered_events.append(event_row)

    # entity_location_pairs = itemset_causality.get_entiry_location_pairs(filtered_events, header)
    #
    # keyword_relations = itemset_causality.get_keyword_relations(entity_location_pairs, filtered_events, header)
    #
    # truthful_relations = itemset_causality.get_truthful_relations(keyword_relations, filtered_events, header)
    #
    # common_goal_relations = itemset_causality.get_common_goal_relations(truthful_relations)
    #
    # causal_chains = itemset_causality.get_causal_chains(common_goal_relations, filtered_events, header)

    # for causal_chain in causal_chains:
    #     print(' > '.join(causal_chain))
    #
    # print('Total chains: %d' % len(causal_chains))
    end_time = timeit.default_timer()
    print('Execution time: %.02f seconds' % (end_time-start_time))
