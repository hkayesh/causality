import pickle
import csv
import os
import time
from glob import glob


class Utilities:

    def read_from_csv(self, file_path):
        """
        Read data from a csv file

        :param file_path: file path
        :return list: list of row elements 
        """
        if not os.path.exists(file_path):
            raise IOError('FILE_NOT_FOUND')

        data = []

        with open(file_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data.append(row)

        return data

    def save_or_append_list_as_csv(self, data_list, file_path):
        """
        Save a python list as csv file

        :param data_list: data list
        :param file_path: file path
        :return: None 
        """

        if os.path.exists(file_path):
            file_mode = 'a'
        else:
            file_mode = 'w'

        with open(file_path, file_mode) as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(data_list)

    def save_or_append_in_csv(self, data_dict, file_path):
        """
        Save results to new file or append if file exists

        :param data_dict: data dictionary
        :param file_path: output file path

        :return: None
        """
        if os.path.exists(file_path):
            with open(file_path, 'a') as resultFile:
                w = csv.DictWriter(resultFile, data_dict.keys())
                w.writerow(data_dict)
        else:
            with open(file_path, 'w') as resultFile:
                w = csv.DictWriter(resultFile, data_dict.keys())
                w.writeheader()
                w.writerow(data_dict)

    def get_file_paths_in_dir(self, dir_path, file_type='xml'):
        """
        get all file path of a type in a directory

        :param dir_path: directory path
        :param file_type: file type e.g. 'xml'
        :return: 
        """
        file_paths = [y for x in os.walk(dir_path) for y in glob(os.path.join(x[0], '*.' + file_type))]
        return file_paths

    def get_todays_dir(self, base_path):
        """
        Get path to the subdirectory with yyyymmdd name format 

        :param base_path: base path
        :return str: subdir path 
        """
        return base_path + time.strftime("%Y%m%d") + '/'

    def read_lines_from_file(self, file_path):
        """
        Read lines as list from a text file

        :param file_path: file path
        :return list: list of lines 
        """
        if not os.path.exists(file_path):
            raise IOError('FILE_NOT_FOUND')

        file = open(file_path, 'r')
        lines = [line.strip() for line in file.readlines()]

        return lines

    def save_list_as_text_file(self, the_list, file_path):
        """
        save a list as a text file

        :param the_list: a list to save 
        :param file_path: file path; should be a text file
        :return: True if saved properly
        """
        try:
            the_file = open(file_path, 'w')

            for item in the_list:
                the_file.write("%s\n" % item)
        except Exception:
            raise IOError("FAILED_TO_SAVE")

        return True

    def pretty_time_delta(self, seconds):
        sign_string = '-' if seconds < 0 else ''
        seconds = abs(int(seconds))
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
        elif hours > 0:
            return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
        elif minutes > 0:
            return '%s%dm%ds' % (sign_string, minutes, seconds)
        else:
            return '%s%ds' % (sign_string, seconds)

    def chunkify_list(self, data_list, items_per_chunk):
        # For item i in a range that is a length of l,
        for i in range(0, len(data_list), items_per_chunk):
            # Create an index range for l of n items:
            yield data_list[i:i + items_per_chunk]