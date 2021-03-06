import datetime
import collections

from utils.utilities import Utilities
from causality_detection.causal_stength_calculator import CausalStrengthCalculator


if __name__ == "__main__":
    event_file_path = 'events.csv'
    utilities = Utilities()
    causal_strength_calculator = CausalStrengthCalculator()

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
    for row in event_rows:
        event_tokens = row[header.index('event_phrases')].split(',')
        meta_info[len(event_tokens)] = 1 if len(event_tokens) not in list(meta_info.keys()) else meta_info[len(event_tokens)] + 1

        # for event_token in event_tokens:
        #     meta_info[event_token] = 1 if event_token not in list(meta_info.keys()) else meta_info[event_token] + 1

        # if len(row[header.index('entities')]) > 0 and row[header.index('locations')] and len(event_tokens) == 1:
        #    event_rows.append(row)
    filtered_events = []
    for event_row in event_rows:
        event_tokens = event_row[header.index('event_phrases')].split(',')
        if len(event_row[header.index('entities')]) > 0 and event_row[header.index('locations')] and len(event_tokens) == 1:
            filtered_events.append(event_row)

    candidate_event_pairs = causal_strength_calculator.get_causality_candidates(event_rows[:10], header)


    causality_scores = []
    for causal_event, effect_event in candidate_event_pairs:
        causal_candidate_phrase = causal_event[header.index('event_phrases')]
        effect_candidate_phrase = effect_event[header.index('event_phrases')]

        causality_score = causal_strength_calculator.get_causality_score(causal_candidate_phrase, effect_candidate_phrase)
        causality_scores.append((causal_candidate_phrase, effect_candidate_phrase, causality_score))

    causality_scores_ordered = sorted(causality_scores, key=lambda x: float(x[2]), reverse=True)


    for row in causality_scores_ordered[:100]:
        if row[2] > 0:
            print(row)

