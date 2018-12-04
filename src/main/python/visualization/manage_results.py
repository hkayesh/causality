import os
import json


class ManageResults(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def save_dictionary_to_file(self, results, result_key):
        data = {}

        if os.path.exists(self.file_path):
            with open(self.file_path) as json_file:

                content = json_file.read()
                if len(content) > 0:
                    data = json.loads(content)

        data[result_key] = results

        with open(self.file_path, 'w') as outfile:
            json.dump(data, outfile)
