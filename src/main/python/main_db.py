from utils.utilities import Utilities
from database.db_operations import DbOperations
from required_files.config import app_config

if __name__ == '__main__':

    db_operations = DbOperations()
    utilities = Utilities()

    items = db_operations.get_all_item()
    data_file = app_config['data_file']
    for item in items:
        utilities.save_or_append_in_csv(item, data_file)
