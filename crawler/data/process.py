#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import re

output_file = open("list_phoenix_20180101_20190314.txt", "w")

soup = BeautifulSoup(open('phoenix_20180101_20190314.txt','r').read(), 'html.parser')
for tag in soup.find_all('a'):
    if tag.contents[0] == u"表示":
        link = tag.attrs["href"]
        date = link.split("/")[1].split(".")[0][3:]
        output_file.write("%s,https://tenhou.net/sc/raw/%s\n" % (date, link))