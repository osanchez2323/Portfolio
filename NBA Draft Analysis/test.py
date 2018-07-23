# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:39:36 2018

@author: osanchez
"""





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

seasons = pd.read_csv('C:/Users/osanchez/Downloads/Seasons_Stats.csv')




years = list(range(1980,2018))
seasons = seasons[seasons['Year'].isin(years)]




seasons['MPG'] = seasons['MP'] / seasons['G']

seasons = seasons[seasons['MPG'] >= 20]





seasons[(seasons['Player'] == 'J.J. Redick') & (seasons['Year'] > 2011)]



seasons = seasons[seasons['Tm'] != 'TOT']
seasons['Player'] = seasons['Player'].str.replace('*','') ### Remove special character





players = pd.read_csv('C:/Users/osanchez/Downloads/player_data.csv')
years_draft = list(range(2010,2017))
players.columns = ['Player', 'Rookie Year', 'Final Year', 'Position', 'Height','Weight','DOB','College']
players = players[players['Rookie Year'].isin(years_draft)]
players['Player'] = players['Player'].str.replace('*','') ### Remove special character







data = seasons[['Year', 'Player','G','MP','MPG','PER', 'TS%', 'FG%', '3P%','2P%', 'eFG%','OWS', 'DWS', 'WS', 'WS/48',
                'USG%','OBPM', 'DBPM', 'BPM', 'VORP']]
data = data[data['Year'].isin(years)]
data['Year'] = data['Year'].astype(object)

data_draft = players.merge(data, on = 'Player')
data_draft['Rookie Year'] = data_draft['Rookie Year'] - 1 ### Set the Year to the correct draft class

f = {'G': ['sum'],'MP': ['sum'],'MPG': ['mean'],'PER': ['mean'], 'TS%': ['mean'], 'FG%': ['mean'], '3P%': ['mean'],'2P%': ['mean'], 
     'eFG%': ['mean'],'OWS': ['mean'], 'DWS': ['mean'], 'WS': ['mean'], 'WS/48': ['mean'],'USG%': ['mean'],
     'OBPM': ['mean'], 'DBPM': ['mean'], 'BPM': ['mean'], 'VORP': ['mean']}
data_draft = data_draft.groupby(['Rookie Year'], as_index = False).agg(f)





per = data_draft.sort_values([('PER','mean')], ascending = False).reset_index()
pos = list(range(len(per['PER','mean'])))

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (15,8))
plt.bar(pos, per['PER','mean'], width = 0.75, alpha = 0.75, label = per['Rookie Year'], edgecolor = 'gray', linewidth = 2)
for i in pos:
    plt.text(pos[i], 6, s = per['Rookie Year'][i],ha='center', va='bottom', rotation = 'vertical',color = 'orange', size = 50)
plt.text(x = -0.68, y = 18, s = 'Total Career PER - 1984 Draft Class lead in efficiency',fontsize = 30, weight = 'bold', alpha = .75)
plt.text(x = -0.68, y = 17, s = 'Total Career PER for all NBA players drafted in 1984, 1996, and 2003',fontsize = 19, alpha = .85)
plt.text(x = -0.68, y = -1.5, s = 'National Basketball Association                                                                                       Source: Basketball-Reference.com ', fontsize = 17,color = '#f0f0f0', backgroundcolor = 'grey')
plt.xticks([],[])
ax.set_ylabel('PER', size = 25)







vorp = data_draft.sort_values([('VORP','mean')], ascending = False).reset_index()
pos1 = list(range(len(vorp['VORP','mean'])))

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (15,8))
plt.bar(pos1, vorp['VORP','mean'], width = 0.75, alpha = 0.75, label = vorp['Rookie Year'], edgecolor = 'gray', linewidth = 2)
for i in pos1:
    plt.text(pos1[i], .6, s = vorp['Rookie Year'][i],ha='center', va='bottom', color = 'orange', size = 50)
plt.text(x = -0.68, y = 2.35, s = 'Total Career VORP - Jordan and friends keep dominating',fontsize = 30, weight = 'bold', alpha = .75)
plt.text(x = -0.68, y = 2.2, s = 'Total Career VORP for all NBA players drafted in 1984, 1996, and 2003',fontsize = 19, alpha = .85)
plt.text(x = -0.7, y = -.15, s = 'National Basketball Association                                                                                       Source: Basketball-Reference.com ', fontsize = 17,color = '#f0f0f0', backgroundcolor = 'grey')
plt.xticks([],[])
ax.set_ylabel('VORP', size = 25)





ws = data_draft.sort_values([('WS/48','mean')], ascending = False).reset_index()
pos = list(range(len(ws['WS/48','mean'])))

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize = (15,8))
plt.bar(pos, ws['WS/48','mean'], width = 0.75, alpha = 0.75, label = ws['Rookie Year'], edgecolor = 'gray', linewidth = 2)
for i in pos:
    plt.text(pos[i], 0.045, s = ws['Rookie Year'][i],ha='center', va='bottom', color = 'orange', size =50)

plt.text(x = -0.7, y = 0.135, s = 'Total Career WS/48 - We get it. 1984 is the best draft class',fontsize = 30, weight = 'bold', alpha = .75)
plt.text(x = -0.7, y = 0.1255, s = 'Total Career WS/48 for all NBA players drafted in 1984, 1996, and 2003',fontsize = 19, alpha = .85)
plt.text(x = -0.71, y = -.012, s = 'National Basketball Association                                                                                       Source: Basketball-Reference.com ', fontsize = 17,color = '#f0f0f0', backgroundcolor = 'grey')
plt.xticks([],[])
ax.set_ylabel('WS/48', size = 25)
















































