from utils.utilities import Utilities
from preprocessing.event_extractor import EventExtractor
import time

if __name__ == '__main__':
    # Original Code
    #
    # utilities = Utilities()
    # event_extraction = EventExtractor()
    # #
    # # event_extraction.save_texts_in_file()
    #
    # events = event_extraction.extract_events()
    #
    # for event in events:
    #     if len(event['event_phrases']) > 0:
    #         utilities.save_or_append_in_csv(event, 'events.csv')

    utilities = Utilities()
    event_extractor = EventExtractor()

    tweet_rows = event_extractor.get_unique_tweets()

    tweet_sentences = event_extractor.get_tweet_sentences(tweet_rows)

    events = event_extractor.extract_events2(tweet_sentences)
    events = sorted(events, key=lambda x: time.strptime(x[0], '%d-%m-%Y %H:%M'))
    utilities.save_or_append_list_as_csv(events, 'events2.csv')

