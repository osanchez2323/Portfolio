# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:31:28 2018

@author: osanchez
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from scipy.stats import norm
from operator import itemgetter
import datetime


seasons = pd.read_csv('C:/Users/osanchez/Downloads/Seasons_Stats.csv')
players = pd.read_csv('C:/Users/osanchez/Downloads/Players.csv')
years = list(range(1990,2018))

seasons = seasons[seasons['Year'].isin(years)]
seasons = seasons[seasons['Tm'] != 'TOT']
seasons['PPG'] = seasons['PTS'] / seasons['G']


test = pd.read_csv('C:/Users/osanchez/Downloads/2017-18_playerBoxScore.csv')
test['G'] = 1
test = test.groupby(['playDispNm'], as_index = False).sum()
test['PPG'] = round(test['playPTS'] / test['G'],1)

seasons.columns
test.columns

data = seasons[['Year', 'Player','G','MP','PER', 'TS%','FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
       '2P', '2PA', '2P%', 'eFG%']]
data = data[data['Year'].isin(years)]
data['Year'] = data['Year'].astype(object)

f = {'G': ['sum'],'MP': ['sum'],'PER': ['mean'], 'TS%': ['mean'],'FG': ['sum'], 'FGA': ['sum'], 'FG%': ['mean'], '3P': ['sum'], '3PA': ['sum'], '3P%': ['mean'],
       '2P': ['sum'], '2PA': ['sum'], '2P%': ['mean'], 'eFG%': ['mean']}
data = data.groupby(['Player'], as_index = False).agg(f)






import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import animation
import io
import base64
from IPython.display import HTML
plt.style.use('fivethirtyeight')
from subprocess import check_output



seasons = seasons.drop(seasons.index[len(seasons) - 1])
seasons = seasons[pd.isnull(seasons.Player) == 0]
seasons['height'] = seasons.Player.apply(lambda x: players.height[players.Player == x].values[0])
seasons['weight'] = seasons.Player.apply(lambda x: players.weight[players.Player == x].values[0])



fig = plt.figure(figsize = (10,10))
ax = plt.axes()


plt.style.use('fivethirtyeight')
def animate(year):
    ax.clear()
    ax.set_xlim([-3,15])
    ax.set_ylim([0,900])
    ax.set_title(str(int(year)))
    ax.set_xlabel('VORP')
    ax.set_ylabel('3PA')
    x = seasons['VORP'][(seasons.Year == year) & (seasons.Pos == 'PG')]
    y = seasons['3PA'][(seasons.Year == year) & (seasons.Pos == 'PG')]
    ax.plot(x,y,'o', color = 'r', markersize = 10, alpha = 0.5)
    x = seasons['VORP'][(seasons.Year == year) & (seasons.Pos == 'SG')]
    y = seasons['3PA'][(seasons.Year == year) & (seasons.Pos == 'SG')]
    ax.plot(x,y,'o', color = 'm', markersize = 10, alpha = 0.5)
    x = seasons['VORP'][(seasons.Year == year) & (seasons.Pos == 'SF')]
    y = seasons['3PA'][(seasons.Year == year) & (seasons.Pos == 'SF')]
    ax.plot(x,y,'o', color = 'b', markersize = 10, alpha = 0.5)
    x = seasons['VORP'][(seasons.Year == year) & (seasons.Pos == 'PF')]
    y = seasons['3PA'][(seasons.Year == year) & (seasons.Pos == 'PF')]
    ax.plot(x,y,'o', color = 'g', markersize = 10, alpha = 0.5)
    x = seasons['VORP'][(seasons.Year == year) & (seasons.Pos == 'C')]
    y = seasons['3PA'][(seasons.Year == year) & (seasons.Pos == 'C')]
    ax.plot(x,y,'o', color = 'y', markersize = 10, alpha = 0.5)
    ax.legend(['PG','SG','SF','PF','C'], loc = 1)


ani = animation.FuncAnimation(fig,animate,seasons.Year.unique().tolist(), interval = 500)
ani.save('animation.html', writer='imagemagick', fps=1)
filename = 'animation.html'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="html" />'''.format(encoded.decode('ascii')))





########


seasons.columns


seasons = pd.read_csv('C:/Users/osanchez/Downloads/Seasons_Stats.csv')
players = pd.read_csv('C:/Users/osanchez/Downloads/Players.csv')
years = list(range(1990,2018))

seasons = seasons[seasons['Year'].isin(years)]
seasons = seasons[seasons['Tm'] != 'TOT']




data = seasons[['Year', 'Player','G','MP','PER', 'TS%', 'FG%', '3P%','2P%', 'eFG%','OWS', 'DWS', 'WS', 'WS/48','USG%','OBPM', 'DBPM', 'BPM', 'VORP']]
data = data[data['Year'].isin(years)]
data['Year'] = data['Year'].astype(object)

f = {'G': ['sum'],'MP': ['sum'],'PER': ['mean'], 'TS%': ['mean'], 'FG%': ['mean'], '3P%': ['mean'],'2P%': ['mean'], 'eFG%': ['mean'],'OWS': ['mean'], 'DWS': ['mean'], 'WS': ['mean'], 
     'WS/48': ['mean'],'USG%': ['mean'],'OBPM': ['mean'], 'DBPM': ['mean'], 'BPM': ['mean'], 'VORP': ['mean']}
data = data.groupby(['Player'], as_index = False).agg(f)









df = pd.read_html('https://en.wikipedia.org/wiki/2010_NBA_draft')[3]
df = df.rename(columns=df.iloc[0]).drop(df.index[0])

df['Player'] = df['Player'].str.replace('+','')
df['Player'] = df['Player'].str.replace('*','')
df['Player'] = df['Player'].str.replace('#','')
df['Player'] = df['Player'].str.replace('~','')
df['Draft'] = 1


Main_data = df.merge(data, on = 'Player')







per = Main_data.sort_values([('PER','mean')], ascending = False).head(10).reset_index()


pos = list(range(len(per['PER','mean'])))




fig, ax = plt.subplots()
plt.bar(pos, per['PER','mean'], width = 0.75, alpha = 0.75, label = per['Player'])
for i in pos:
    plt.text(pos[i], 0.5, s = per['Player'][i],ha='center', va='bottom', rotation = 'vertical', color = 'white')






vorp = Main_data.sort_values([('VORP','mean')], ascending = False).head(10).reset_index()

fig, ax = plt.subplots()
plt.bar(pos, vorp['VORP','mean'], width = 0.75, alpha = 0.75, label = vorp['Player'])
for i in pos:
    plt.text(pos[i], 0.5, s = vorp['Player'][i],ha='center', va='bottom', rotation = 'vertical', color = 'orange')



ts = Main_data[Main_data['MP','sum'] > 5000]
ts = ts.sort_values([('TS%','mean')], ascending = False).head(10).reset_index()


fig, ax = plt.subplots()
plt.bar(pos, ts['TS%','mean'], width = 0.75, alpha = 0.75, label = ts['Player'])
for i in pos:
    plt.text(pos[i], 0.05, s = ts['Player'][i],ha='center', va='bottom', rotation = 'vertical', color = 'orange')


































