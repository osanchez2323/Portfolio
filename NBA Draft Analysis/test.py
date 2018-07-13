# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:39:36 2018

@author: osanchez
"""

seasons = pd.read_csv('Seasons_Stats.csv')
players = pd.read_csv('player_data.csv')
players.columns = ['Player', 'Rookie Year', 'Final Year', 'Position', 'Height','Weight','DOB','College']


years = list(range(1980,2018))
years_draft = [1985, 1997, 2004]

seasons = seasons[seasons['Year'].isin(years)]
seasons = seasons[seasons['MP'] > 500]
players = players[players['Rookie Year'].isin(years_draft)]
seasons = seasons[seasons['Tm'] != 'TOT']




data = seasons[['Year', 'Player','G','MP','PER', 'TS%', 'FG%', '3P%','2P%', 'eFG%','OWS', 'DWS', 'WS', 'WS/48','USG%','OBPM', 'DBPM', 'BPM', 'VORP']]
data = data[data['Year'].isin(years)]
data['Year'] = data['Year'].astype(object)
data['Player'] = data['Player'].str.replace('*','')

data_draft = players.merge(data, on = 'Player')



f = {'G': ['sum'],'MP': ['sum'],'PER': ['mean'], 'TS%': ['mean'], 'FG%': ['mean'], '3P%': ['mean'],'2P%': ['mean'], 'eFG%': ['mean'],'OWS': ['mean'], 'DWS': ['mean'], 'WS': ['mean'], 
     'WS/48': ['mean'],'USG%': ['mean'],'OBPM': ['mean'], 'DBPM': ['mean'], 'BPM': ['mean'], 'VORP': ['mean']}


data_1984 = data_1984.groupby(['Draft'], as_index = False).agg(f)
data_1996 = data_1996.groupby(['Draft'], as_index = False).agg(f)
data_2003 = data_2003.groupby(['Draft'], as_index = False).agg(f)

combine = [data_1984, data_1996, data_2003]







for item in combine:
    item['Player'] = item['Player'].str.replace('+','')
    item['Player'] = item['Player'].str.replace('*','')
    item['Player'] = item['Player'].str.replace('#','')
    item['Player'] = item['Player'].str.replace('~','')
    item['Player'] = item['Player'].str.replace('^','')
    item['Player'] = item['Player'].str.replace('>','')
    item['Player'] = item['Player'].str.replace('\d+','')
    item['Player'] = item['Player'].str.replace('[','')
    item['Player'] = item['Player'].str.replace(']','')
    item['Player'] = item['Player'].str.replace('Hakeem Olajuwon›','Hakeem Olajuwon')
    item['Player'] = item['Player'].str.strip()








data_1984 = df1.merge(data, on = 'Player')
data_1996 = df2.merge(data, on = 'Player')
data_2003 = df3.merge(data, on = 'Player')


data_1984 = data_1984.groupby(['Draft'], as_index = False).sum()

f = {'G': ['sum'],'MP': ['sum'],'PER': ['mean'], 'TS%': ['mean'], 'FG%': ['mean'], '3P%': ['mean'],'2P%': ['mean'], 'eFG%': ['mean'],'OWS': ['mean'], 'DWS': ['mean'], 'WS': ['mean'], 
     'WS/48': ['mean'],'USG%': ['mean'],'OBPM': ['mean'], 'DBPM': ['mean'], 'BPM': ['mean'], 'VORP': ['mean']}


data_draft = data_draft.groupby(['Rookie Year'], as_index = False).agg(f)

combine = pd.concat(combine)



per = data_draft.sort_values([('PER','mean')], ascending = False).reset_index()
pos = list(range(len(per['PER','mean'])))


fig, ax = plt.subplots()
plt.bar(pos, per['PER','mean'], width = 0.75, alpha = 0.75, label = per['Rookie Year'])
for i in pos:
    plt.text(pos[i], 0.5, s = per['Rookie Year'][i],ha='center', va='bottom', rotation = 'vertical', color = 'white')











