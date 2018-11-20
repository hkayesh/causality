import os
import json


class ManageResults(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def save_dictionary_to_file(self, data_dict=None, label=None):
        data = {}

        if os.path.exists(self.file_path):
            with open(self.file_path) as json_file:

                content = json_file.read()
                if len(content) > 0:
                    data = json.loads(content)

        for key in data_dict.keys():
            if key in data.keys():
                data[key][label] = data_dict[key]
            else:
                data[key] = {label: data_dict[key]}

        with open(self.file_path, 'w') as outfile:
            json.dump(data, outfile)
