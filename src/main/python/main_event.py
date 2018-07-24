from utils.utilities import Utilities
from preprocessing.event_extractor import EventExtractor

if __name__ == '__main__':
    utilities = Utilities()
    event_extraction = EventExtractor()
    #
    # event_extraction.save_texts_in_file()

    events = event_extraction.extract_events()

    for event in events:
        if len(event['event_phrases']) > 0:
            utilities.save_or_append_in_csv(event, 'events.csv')
