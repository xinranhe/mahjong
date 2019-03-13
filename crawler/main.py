#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from bs4 import BeautifulSoup
import datetime
from multiprocessing import Pool
import requests
import os
import sys

NUM_THREADS = 8
INPUT_FILE = "crawler/data/list_phoenix_20180101_20190314.txt"
USER_AGENT = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Mobile Safari/537.36'

def parse_list_page(content):
    log_ids = []
    for line in content.split("<br>"):
        fields = line[:-1].split("|")
        if len(fields) < 4:
            continue
        if u"四鳳" in fields[2]:
            soup = BeautifulSoup(fields[3], 'html.parser')
            tag = soup.find_all('a')[0]
            log_ids.append(tag.attrs["href"].split("?log=")[1])
    return log_ids

def download_log(log_id, folder):
    print "Download %s/%s" % (folder, log_id)
    f = open("%s/%s.txt" % (folder, log_id), "w")
    link = "https://tenhou.net/3/mjlog2xml.cgi?%s" %  log_id
    response = requests.get(link, headers={'host': 'tenhou.net', 
                                           'referer': 'http://tenhou.net/3/?log=%s' % log_id,
                                           'user-agent': USER_AGENT})
    f.write(response.text)
    f.close()

def process(to_process):
    date, link = to_process
    # create folder
    folder_path = "%s/%s" % (known_args.output_data_dir, date)
    if not os.path.exists(folder_path):
         os.makedirs(folder_path)

    # get main page
    response = requests.get(link, headers={'host': 'tenhou.net', 
                                           'referer': 'https://tenhou.net/sc/raw/?old',
                                           'user-agent': USER_AGENT})
    log_ids = parse_list_page(response.text)
    print "Get %s links for %s" % (len(log_ids), date)
    for log_id in log_ids:
        download_log(log_id, folder_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--start_date', default='')
    parser.add_argument('--end_date', default='')
    parser.add_argument('--output_data_dir', default='')
    known_args, _ = parser.parse_known_args(sys.argv)
    
    process_list = []
    for line in open(INPUT_FILE, "r"):
        date, link = line[:-1].split(",")
        if date >= known_args.start_date and date <= known_args.end_date:
            process_list.append((date, link))

    print process_list
    # multithread crawlering on day level
    p = Pool(NUM_THREADS)
    p.map(process, process_list)
