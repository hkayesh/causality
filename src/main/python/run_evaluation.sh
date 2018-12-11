#!/usr/bin/env bash

python main_evaluation.py --key cnet_wiki_exp_0; python main_evaluation.py --key cnet_wiki_exp_1; python main_evaluation.py --key cnet_wiki_exp_2; python main_evaluation.py --key cnet_wiki_exp_3; python main_evaluation.py --key cnet_wiki_exp_4; python main_evaluation.py --key cnet_wiki_exp_5;

python main_evaluation.py --key cnet_news_exp_0; python main_evaluation.py --key cnet_news_exp_1; python main_evaluation.py --key cnet_news_exp_2; python main_evaluation.py --key cnet_news_exp_3; python main_evaluation.py --key cnet_news_exp_4; python main_evaluation.py --key cnet_news_exp_5;

python main_evaluation.py --key luo_threshold_10;

python main_evaluation.py --key sasaki_threshold_10;

python main_evaluation.py --visualize yes;
