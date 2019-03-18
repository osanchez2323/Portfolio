# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:43:51 2019

@author: osanchez
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Basic NBA Stats


year = 2019
url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year)
# this is the html from the given url
html = urlopen(url)
soup = BeautifulSoup(html)
type(soup)
soup.findAll('tr', limit=2)
column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
column_headers = column_headers[1:]
data_rows = soup.findAll('tr')[2:]
type(data_rows)
player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]






basic_2019 = pd.DataFrame(player_data, columns=column_headers)


year = 2019
url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)
# this is the html from the given url
html = urlopen(url)
soup = BeautifulSoup(html)
type(soup)
soup.findAll('tr', limit=2)
column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
column_headers = column_headers[1:]
data_rows = soup.findAll('tr')[2:]
type(data_rows)
player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]



advanced_2019 = pd.DataFrame(player_data, columns=column_headers)





basic_2019 = basic_2019[basic_2019['Player'].notnull()]

basic_2019 = basic_2019.convert_objects(convert_numeric = True)

basic_2019 = basic_2019[:].fillna(0)

basic_2019 = basic_2019.drop_duplicates(['Player'], keep = 'first')







advanced_2019 = advanced_2019[advanced_2019['Player'].notnull()]

advanced_2019 = advanced_2019.convert_objects(convert_numeric = True)

advanced_2019 = advanced_2019[:].fillna(0)

advanced_2019 = advanced_2019.drop_duplicates(['Player'], keep = 'first')



stats = pd.merge(basic_2019, advanced_2019, on = 'Player')






































