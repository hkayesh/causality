import numpy as np
from statistics import median
from dateutil import parser
from causality_detection.causal_stength_calculator import CausalStrengthCalculator


class ItemsetCausality:

    def __init__(self):
        self.causal_strength_calculator = CausalStrengthCalculator()

    def get_entiry_location_pairs(self, events, header):
        entities = list(set([filtered_event[header.index('entities')] for filtered_event in events]))
        # TODO: split entities

        locations = list(set([filtered_event[header.index('locations')] for filtered_event in events]))
        # TODO: split locations

        entity_location_pairs = []
        for entity in entities:
            for location in locations:
                if entity != location:
                    entity_location_pairs.append((entity, location))

        return entity_location_pairs

    def get_keyword_relations(self, rules, events, event_header):

        keyword_relations = []

        for rule in rules:
            left_hand_side = rule[0]
            right_hand_side = rule[1]

            left_hand_side_keywords = []
            right_hand_side_keywords = []

            for event in events:
                entity_and_location = [event[event_header.index('entities')], event[event_header.index('locations')]]
                event_phrase = event[event_header.index('event_phrases')].strip()

                if left_hand_side in entity_and_location and event_phrase not in right_hand_side_keywords:
                    left_hand_side_keywords.append(event_phrase)

                if right_hand_side in entity_and_location and event_phrase not in left_hand_side_keywords:
                    right_hand_side_keywords.append(event_phrase)

            keyword_rule = (list(set(left_hand_side_keywords)), list(set(right_hand_side_keywords)))

            if len(left_hand_side_keywords) > 0 and len(right_hand_side_keywords) > 0 and keyword_rule not in keyword_relations:
                keyword_relations.append(keyword_rule)

        # for keyword_relation in keyword_relations:
        #     print(keyword_relation)

        return keyword_relations

    def get_events_by_keywords(self, keyword, events, event_header):
        keyword_events = []
        for event in events:
            if keyword == event[event_header.index('event_phrases')] and event not in keyword_events:
                keyword_events.append(event)

        return keyword_events

    def get_truthful_relations(self, keyword_relations, events, event_header):
        truthful_relations = []
        for keyword_relation in keyword_relations:
            lhs_keywords = keyword_relation[0]
            rhs_keywords = keyword_relation[1]

            rule_level_counter = 0

            for lhs_keyword in lhs_keywords:
                lhs_keyword_events = self.get_events_by_keywords(lhs_keyword, events, event_header)
                for rhs_keyword in rhs_keywords:
                    rhs_keyword_events = self.get_events_by_keywords(rhs_keyword, events, event_header)
                    subrule_level_counter = 0
                    for lhs_keyword_event in lhs_keyword_events:
                        lhs_keyword_event_time = parser.parse(lhs_keyword_event[event_header.index('event_time')])
                        for rhs_keyword_event in rhs_keyword_events:
                            rhs_keyword_event_time = parser.parse(rhs_keyword_event[event_header.index('event_time')])

                            if lhs_keyword_event_time > rhs_keyword_event_time:
                                subrule_level_counter += 1
                    rule_level_counter += subrule_level_counter/(len(lhs_keyword_events)*len(rhs_keyword_events))

            truthful_relations_value = (rule_level_counter/(len(lhs_keywords)*len(rhs_keywords))) * 100
            if truthful_relations_value > 80:
                truthful_relations.append(keyword_relation)

        # for truthful_relation in truthful_relations:
        #     print(truthful_relation)

        return truthful_relations

    def get_common_goal_relations(self, relations):
        common_goal_relations = {}
        sub_relations = []
        for relation in relations:
            lhs_keywords = relation[0]
            rhs_keywords = relation[1]

            for lhs_keyword in lhs_keywords:
                for rhs_keyword in rhs_keywords:
                    causal_strength = self.causal_strength_calculator.get_causal_strength(lhs_keyword, rhs_keyword)
                    sub_relations.append((lhs_keyword, rhs_keyword, causal_strength))

        median_score = np.percentile(np.array(list(set([sub_relation[2] for sub_relation in sub_relations]))), 80)

        common_sense_relations = []
        for sub_relation in sub_relations:
            if sub_relation[2] > median_score:
                common_sense_relations.append((sub_relation[0], sub_relation[1]))

        for common_sense_relation in common_sense_relations:
            causal_keyword = common_sense_relation[0]
            effect_keyword = common_sense_relation[1]
            if effect_keyword not in list(common_goal_relations.keys()):
                common_goal_relations[effect_keyword] = [causal_keyword]
            elif effect_keyword in list(common_goal_relations.keys()) and causal_keyword not in common_goal_relations[effect_keyword]:
                common_goal_relations[effect_keyword].append(causal_keyword)

        # for common_goal_relation in common_goal_relations.keys():
        #     print(common_goal_relation)
        #     print(common_goal_relations[common_goal_relation])
        # print(len(list(common_goal_relations.keys())))
        # print(median_score)
        return common_goal_relations

    def get_causal_chains(self, relations, events, event_header):
        causal_chains = []
        for goal in relations.keys():
            cause_keywords = relations[goal]
            event_first_occurrences = {}
            for cause_keyword in cause_keywords:
                cause_keyword_events = self.get_events_by_keywords(cause_keyword, events, event_header)
                for cause_keyword_event in cause_keyword_events:
                    event_time = parser.parse(cause_keyword_event[event_header.index('event_time')]).timestamp()
                    if cause_keyword not in list(event_first_occurrences.keys()):
                        event_first_occurrences[cause_keyword] = event_time
                    else:
                        if event_time < event_first_occurrences[cause_keyword]:
                            event_first_occurrences[cause_keyword] = event_time

            sorted_event_first_occurrences = [(k, event_first_occurrences[k]) for k in sorted(event_first_occurrences, key=event_first_occurrences.get)]

            causal_chain = []
            for sorted_event_first_occurrence in sorted_event_first_occurrences:
                causal_chain.append(sorted_event_first_occurrence[0])

            causal_chain.append(goal)
            causal_chains.append(causal_chain)

        return causal_chains

