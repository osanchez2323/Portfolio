import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

### Load Data
all_salaries = pd.read_csv('.../Salaries.csv')
all_team = pd.read_csv('.../Teams.csv')
all_batting = pd.read_csv('...Batting.csv')

### Organize data for Steroid Era Stats
steroid_years = list(range(1990,2004))
steroid = all_batting[['playerID','yearID','teamID','G','AB','R','H', 'HR', 'RBI','BB','2B', '3B', 'HBP', 'SF']]
steroid = steroid[steroid['yearID'].isin(steroid_years)]

steroid = steroid.groupby(['playerID', 'yearID'], as_index = False).sum()
steroid = steroid[steroid['G'] > 100]
steroid = steroid.groupby(['playerID'], as_index = False).sum()
steroid = steroid.drop(columns = ['yearID'])

## Calculate addition baseball stats
steroid['AVG'] = round((steroid['H'] / steroid['AB']),3)
steroid['TB'] = ((steroid['2B']*2) + (steroid['3B']*3) + (steroid['HR']*4) + (steroid['H'] - steroid['HR'] - steroid['3B'] - steroid['2B']))
steroid['OBP'] = round((steroid['H'] + steroid['BB'] + steroid['HBP']) / (steroid['AB'] + steroid['BB'] + steroid['HBP'] + steroid['SF']), 3)
steroid['SLG'] = round((steroid['TB'] / steroid['AB']), 3)
steroid['OPS'] = round((steroid['OBP'] + steroid['SLG']), 3)


## Replace missing data with mean of stat
steroid['OBP'] = steroid['OBP'].replace(np.nan, round(steroid['OBP'].mean(),3))
steroid['SLG'] = steroid['SLG'].replace(np.nan, round(steroid['SLG'].mean(),3))
steroid['OPS'] = steroid['OPS'].replace(np.nan, round(steroid['OPS'].mean(),3))
steroid['HBP'] = steroid['HBP'].replace(np.nan, round(steroid['HBP'].mean(),0))
steroid['SF'] = steroid['SF'].replace(np.nan, round(steroid['SF'].mean(),0))






### Organize data for Modern Era Stats
modern_years = list(range(2004,2017))
modern = all_batting[['playerID','yearID','teamID','G','AB','R','H', 'HR', 'RBI','BB','2B', '3B', 'HBP', 'SF']]
modern = modern[modern['yearID'].isin(modern_years)]

modern = modern.groupby(['playerID', 'yearID'], as_index = False).sum()
modern = modern[modern['G'] > 100]
modern = modern.groupby(['playerID'], as_index = False).sum()
modern = modern.drop(columns = ['yearID'])

## Calculate addition baseball stats

modern['AVG'] = round((modern['H'] / modern['AB']),3)
modern['TB'] = ((modern['2B']*2) + (modern['3B']*3) + (modern['HR']*4) + (modern['H'] - modern['HR'] - modern['3B'] - modern['2B']))
modern['OBP'] = round((modern['H'] + modern['BB'] + modern['HBP']) / (modern['AB'] + modern['BB'] + modern['HBP'] + modern['SF']), 3)
modern['SLG'] = round((modern['TB'] / modern['AB']), 3)
modern['OPS'] = round((modern['OBP'] + modern['SLG']), 3)


## Replace missing data with mean of stat
modern['OBP'] = modern['OBP'].replace(np.nan, round(modern['OBP'].mean(),3))
modern['SLG'] = modern['SLG'].replace(np.nan, round(modern['SLG'].mean(),3))
modern['OPS'] = modern['OPS'].replace(np.nan, round(modern['OPS'].mean(),3))
modern['HBP'] = modern['HBP'].replace(np.nan, round(modern['HBP'].mean(),0))
modern['SF'] = modern['SF'].replace(np.nan, round(modern['SF'].mean(),0))






from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot,widgetbox
from bokeh.models.widgets import Tabs,Panel,DataTable, DateFormatter, TableColumn,NumberFormatter,Select
from bokeh.models import NumeralTickFormatter

output_notebook()

top_steroid = steroid.sort_values(['R'], ascending=False)
top_steroid = top_steroid.head(100)


## Set Up Data Sources
source = ColumnDataSource(top_steroid)

hover = HoverTool(
            tooltips = [
                ('Year', '@{yearID}'),
                ('Player','@{playerID}'),
                ('AVG', '@{AVG}{0.03f}'),
                ('HR', '@{HR}{0,0}'),
                ('Runs', '@{R}{0,0}'),
                ('OBP', '@{OBP}{0.03f}'),
                ('SLG', '@{SLG}{0.03f}'),
                ('OPS', '@{OPS}{0.03f}'),
                ])

# Set Up Plots

### Bokeh for Steroid
plot = figure(plot_width = 800, plot_height = 500, tools = [hover, 'box_zoom', 'pan','reset'], active_drag = 'box_zoom', title = 'Top Power Hitters (1990-2003)')
plot.circle(x = 'HR', y = 'R', size = 7, source=source)

plot.title.align = 'center'
plot.title.text_font_size = '20pt'

plot.xaxis.axis_label = 'Home Runs'
plot.xaxis.axis_label_standoff = 20
plot.xaxis.axis_label_text_font_style = 'normal'
plot.xaxis.axis_label_text_font_size = '12pt'


plot.yaxis.axis_label = 'Runs'
plot.yaxis.axis_label_standoff = 20
plot.yaxis.axis_label_text_font_style = 'normal'
plot.yaxis.axis_label_text_font_size = '12pt'

show(plot);



### Matplotlib Plot for Steroid
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('fivethirtyeight')
top_steroid.plot(kind = 'scatter', x = 'HR', y = 'R', s= 50, alpha = 1, figsize = (12,8))
plt.title('Top Power Hitters (1990-2003)', size = 25)
plt.xlabel('Home Runs', size = 15)
plt.ylabel('Runs', size = 15)
plt.axhline(y = 140, color = 'red', linewidth = 2, alpha = 0.7)
plt.axhline(y = 115, color = 'black', linewidth = 1.3, alpha = 0.7)

# The signature bar
plt.text(x = 0, y = 107,
    s = '                                                                                                              Source: Sean Lahman\'s Baseball Database   ',
    fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')

plt.show();







top_modern = modern.sort_values(['R'], ascending=False)
top_modern = top_modern.head(100)


## Set Up Data Sources
source1 = ColumnDataSource(top_modern)

hover1 = HoverTool(
            tooltips = [
                ('Year', '@{yearID}'),
                ('Player','@{playerID}'),
                ('AVG', '@{AVG}{0.03f}'),
                ('HR', '@{HR}{0,0}'),
                ('Runs', '@{R}{0,0}'),
                ('OBP', '@{OBP}{0.03f}'),
                ('SLG', '@{SLG}{0.03f}'),
                ('OPS', '@{OPS}{0.03f}'),
                ])

# Set Up Plots

### Bokeh for Modern
plot1 = figure(plot_width = 800, plot_height = 500, tools = [hover1, 'box_zoom', 'pan','reset'], active_drag = 'box_zoom', title = 'Top Power Hitters (2004-2016)')
plot1.circle(x = 'HR', y = 'R', size = 7, source=source1)

plot1.title.align = 'center'
plot1.title.text_font_size = '20pt'

plot1.xaxis.axis_label = 'Home Runs'
plot1.xaxis.axis_label_standoff = 20
plot1.xaxis.axis_label_text_font_style = 'normal'
plot1.xaxis.axis_label_text_font_size = '12pt'


plot1.yaxis.axis_label = 'Runs'
plot1.yaxis.axis_label_standoff = 20
plot1.yaxis.axis_label_text_font_style = 'normal'
plot1.yaxis.axis_label_text_font_size = '12pt'

show(plot1);






### Matplotlib Plot for Modern
style.use('fivethirtyeight')
top_modern.plot(kind = 'scatter', x = 'HR', y = 'R', s= 50, alpha = 1, figsize = (12,8))
plt.title('Top Power Hitters (2004-2016)', size = 25)
plt.xlabel('Home Runs', size = 15)
plt.ylabel('Runs', size = 15)
plt.axhline(y = 140, color = 'red', linewidth = 2, alpha = 0.7)
plt.axhline(y = 110, color = 'black', linewidth = 1.3, alpha = 0.7)

# The signature bar
plt.text(x = 0, y = 107,
    s = '                                                                                                              Source: Sean Lahman\'s Baseball Database   ',
    fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')

plt.show();












### Create function that returns scatter plot for a list of baseball stats compared to Runs. Also returns correlation of each stat vs Runs
def runs_analysis(data, statistics):
    for stat in statistics:
        t = 'R'
        x = data[stat]
        y = data[t]
        plt.figure()
        plt.scatter(x,y)
        plt.title('{} vs {}'.format(stat,t))
        plt.ylabel('Runs')
        plt.xlabel('{}'.format(stat))
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), 'r--')
        print('The correlation between average {} and {} is {:0.3f}'.format(stat, t, data.corr()[stat][t]))



stats = ['AVG','OPS','OBP', 'SLG']

runs_analysis(steroid, stats)

runs_analysis(modern, stats)






# Bonus: Machine Learning
from sklearn.model_selection import train_test_split
from sklearn import linear_model





### Runs predictions for modern

steroid_sample = steroid[['H', 'HR', 'RBI','AVG','OPS','OBP', 'SLG']]
steroid_runs = np.array(steroid['R'])

steroid_train, steroid_test, steroid_runs_train, steroid_runs_test = train_test_split(steroid_sample, steroid_runs, test_size = 0.2)


# fit a model
lm1 = linear_model.LinearRegression()

model1 = lm1.fit(steroid_train, steroid_runs_train)
predictions1 = lm1.predict(steroid_test)


model1.score(steroid_test, steroid_runs_test)









### Runs predictions for modern

modern_sample = modern[['H', 'HR', 'RBI','AVG','OPS','OBP', 'SLG']]
modern_runs = np.array(modern['R'])

modern_train, modern_test, modern_runs_train, modern_runs_test = train_test_split(modern_sample, modern_runs, test_size = 0.2)


# fit a model
lm2 = linear_model.LinearRegression()

model2 = lm2.fit(modern_train, modern_runs_train)
predictions2 = lm2.predict(modern_test)


model2.score(modern_test, modern_runs_test)






