from background_knowledge.background_data_collection import BackgroundDataCollection

if __name__ == "__main__":
    background_data_collection = BackgroundDataCollection()
    background_data_collection.set_data_source_file('required_files/background_data_source.txt')

    background_data_collection.collect_data('20091101', '20091130')

