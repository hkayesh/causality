import os
from utils.utilities import Utilities
from dateutil import parser
import subprocess
import os

class BackgroundDataCollection:

    def __init__(self):
        self.data_source_file = None
        self.utilities = Utilities()

    def set_data_source_file(self, source_file):
        self.data_source_file = source_file

    def remove_out_of_range_historic_urls(self, urls, date_from, date_to):
        """
        Remove out of range urls

        :param urls: list of urls
        :param date_from: date from
        :param date_to: date to
        :return: list of in the range urls
        """
        in_range_urls = []

        try:
            date_from = parser.parse(str(date_from))
            date_to = parser.parse(str(date_to))
        except ValueError:
            raise Exception("Invalid date range. Please input date in yyyymmdd format")

        for url in urls:
            if len(url) > 43:
                date_str = url[28:42]

                url_time = parser.parse(date_str)

                if date_from <= url_time <= date_to:
                    in_range_urls.append(url)

        return in_range_urls

    def collect_data(self, date_from, date_to):
        """
        Run the whole workflow for historical article collection within a range

        :param date_from: date from
        :param date_to: date to
        :return: list of articles
        """

        # os.makedirs(self.articles_base_dir, exist_ok=True)
        try:
            parser.parse(str(date_from))
            parser.parse(str(date_to))
        except ValueError:
            print("Invalid date format. Please provide date in yyyymmdd format.")
            return

        source_urls = self.utilities.read_lines_from_file(self.data_source_file)
        new_file_count = 0
        for source_url in source_urls:

            url_str = str(subprocess.run(
                ['waybackpack', source_url, '--list', '--from-date', str(date_from), '--to-date', str(date_to)],
                stdout=subprocess.PIPE).stdout.decode('utf-8'))
            urls = url_str.splitlines()

            print(urls)
            exit()

        #     # remove out of range urls returned by waybackpack
        #     urls = self.remove_out_of_range_historic_urls(urls, date_from, date_to)
        #     articles = []
        #     for url in urls:
        #         articles_from_url = self.get_articles_from_rss_url(url, existing_articles_urls)
        #         if len(articles_from_url) > 0:
        #             articles = articles + articles_from_url
        #
        #     if len(articles) < 1:
        #         continue
        #
        #     for article in articles:
        #         file_path = self.get_file_path(destination_folder)
        #         xml_string = self.xml_builder.build_xml(article)
        #         self.xml_builder.save_article_xml(xml_string, file_path)
        #         new_file_count += 1
        #
        # print("Task Completed: %d new articles saved." % new_file_count)
        # print("Total articles: %d\n" % self.utilities.count_files_in_dir(self.articles_base_dir))
