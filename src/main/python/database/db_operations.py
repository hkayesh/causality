import json

from pymongo import MongoClient
from required_files.config import db_config


class DbOperations:

    def __init__(self):
        db_client = MongoClient(db_config['host'], db_config['port'])
        db = db_client[db_config['database']]
        self.db_collection = db[db_config['collection']]

    def get_all_item(self):
        rows = self.db_collection.find({}, {'_id': 0 })

        items = [row for row in rows]
        return items
