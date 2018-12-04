import os
import time
import math
import networkx as nx
from datetime import datetime
from multiprocessing import Manager, Process, cpu_count

from utils.utilities import Utilities
from preprocessing.causal_net_generator import CausalNetGenerator
from preprocessing.causal_net_generator import CausalNetGeneratorFromNews


def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def dispatch_jobs(data, job_number, tokens):
    total = len(data)
    chunk_size = math.ceil(total / job_number)
    slices = chunks(data, chunk_size)
    jobs = []

    for slice in slices:
        job = Process(target=do_job, args=(slice, tokens))
        jobs.append(job)
    for job in jobs:
        job.start()
        job.join()


def do_job(articles, tokens):
    causal_pair_tokens = causal_net_generator.get_all_causal_pair_tokens(articles)

    tokens += causal_pair_tokens


if __name__ == '__main__':
    start = time.time()
    print("\nJob started at %s" % datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'))
    causal_net_generator = CausalNetGenerator()
    causal_net_generator_from_news = CausalNetGeneratorFromNews()
    utilities = Utilities()
    manager = Manager()

    ## Generate causal net from wikipedia articles

    tokens = manager.list()
    num_threads = cpu_count()-1
    number = 1000000
    offset = 0
    print("Number: %d and offset %d" % (number, offset))

    graph_path = 'causal_net.pickle'

    articles = causal_net_generator.get_articles(number=number, offset=offset)
    dispatch_jobs(articles, num_threads, tokens)

    graph = nx.read_gpickle(graph_path) if os.path.exists(graph_path) else None

    causal_net = causal_net_generator.create_or_update_directed_causal_graph(tokens, graph=graph)

    nx.write_gpickle(causal_net, graph_path)

    net = nx.read_gpickle(graph_path)


    ## Generate causal net from news articles

    tokens = manager.list()
    num_threads = cpu_count() - 1
    number = 1000000
    offset = 0
    print("Number: %d and offset %d" % (number, offset))

    graph_path = 'causal_net_news.pickle'

    articles = causal_net_generator_from_news.get_articles(number=number, offset=offset)
    dispatch_jobs(articles, num_threads, tokens)

    graph = nx.read_gpickle(graph_path) if os.path.exists(graph_path) else None

    causal_net = causal_net_generator.create_or_update_directed_causal_graph(tokens, graph=graph)

    nx.write_gpickle(causal_net, graph_path)

    net = nx.read_gpickle(graph_path)

    end = time.time()
    print("Job finished at %s" % datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S'))
    print("Elapsed time: %s" % (utilities.pretty_time_delta(end-start)))









