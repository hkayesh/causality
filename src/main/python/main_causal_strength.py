import datetime

from utils.utilities import Utilities
from causality_detection.causal_stength_calculator import CausalStrengthCalculator


def get_causality_candidates(events_rows):

    chronological_event_rows = sorted(events_rows, key=lambda x: datetime.datetime.strptime(x[3], '%d-%m-%Y %H:%M'))

    causal_candidates = []
    for i, causal_event_row in enumerate(chronological_event_rows):
        for j, effect_event_row in enumerate(chronological_event_rows[i+1:]):
            if causal_event_row[4] != effect_event_row[4]:
                causal_candidates.append((causal_event_row, effect_event_row))

    return causal_candidates


if __name__ == "__main__":
    event_file_path = 'events.csv'
    utilities = Utilities()
    causal_strength_calculator = CausalStrengthCalculator()

    rows = utilities.read_from_csv(event_file_path)
    header = rows[0]
    del rows[0]

    event_rows = []
    for row in rows:
        event_tokens = row[header.index('event_phrases')].split(',')
        if len(row[header.index('entities')]) > 0 and row[header.index('locations')] and len(event_tokens) == 1:
            event_rows.append(row)

    candidate_event_pairs = get_causality_candidates(event_rows[:10])

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

