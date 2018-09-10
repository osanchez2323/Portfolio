
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as matplot
import matplotlib.gridspec as gridspec


### Data Cleaning


data = pd.read_csv('2016 School Explorer.csv')
data = data[pd.notnull(data['Student Achievement Rating'])]



percentage_features = ['Percent of Students Chronically Absent','Rigorous Instruction %','Collaborative Teachers %',
                       'Supportive Environment %','Effective School Leadership %','Strong Family-Community Ties %',
                       'Trust %','Student Attendance Rate','Percent ELL','Percent Asian','Percent Black',
                       'Percent Hispanic','Percent White','Percent Black / Hispanic']

def p2f(x):
    return float(x.strip('%')) / 100

for i in percentage_features:
    data[i] = data[i].astype(str).apply(p2f)





data['School Income Estimate'] = data['School Income Estimate'].str.replace(',', '')
data['School Income Estimate'] = data['School Income Estimate'].str.replace('$', '')
data['School Income Estimate'] = data['School Income Estimate'].str.replace(' ', '')
data['School Income Estimate'] = data['School Income Estimate'].astype(float)


x = data[['Economic Need Index','School Income Estimate']]

x = x.groupby(['Economic Need Index'], as_index = False).dropna().mean()


data.loc[data['Economic Need Index'] > 0.9, 'Economic Need'] = 100000






rating_features = ['Supportive Environment Rating','Rigorous Instruction Rating','Collaborative Teachers Rating',
                   'Effective School Leadership Rating','Strong Family-Community Ties Rating','Student Achievement Rating',
                  'Trust Rating']



for i in rating_features:
    data[i] = data[i].fillna(data[i].mode()[0])



data['Supportive Environment Rating'] = data['Supportive Environment Rating'].fillna(data['Supportive Environment Rating'].mode()[0])

data['Rigorous Instruction Rating'] = data['Rigorous Instruction Rating'].fillna(data['Rigorous Instruction Rating'].mode()[0])

data['Collaborative Teachers Rating'] = data['Collaborative Teachers Rating'].fillna(data['Collaborative Teachers Rating'].mode()[0])


data['Effective School Leadership Rating'] = data['Effective School Leadership Rating'].fillna(data['Effective School Leadership Rating'].mode()[0])

data['Strong Family-Community Ties Rating'] = data['Strong Family-Community Ties Rating'].fillna(data['Strong Family-Community Ties Rating'].mode()[0])

data['Student Achievement Rating'] = data['Student Achievement Rating'].fillna(data['Student Achievement Rating'].mode()[0])

data['Trust Rating'] = data['Trust Rating'].fillna(data['Trust Rating'].mode()[0])



avg_features = ['School Income Estimate','Economic Need Index','Student Attendance Rate','Average ELA Proficiency',
                'Average Math Proficiency']


for i in avg_features:
    data[i] = data[i].fillna(data[i].dropna().mean())



data['School Income Estimate'] = data['School Income Estimate'].fillna(data['School Income Estimate'].dropna().mean())
data['Economic Need Index'] = data['Economic Need Index'].fillna(data['Economic Need Index'].dropna().mean())
data['Student Attendance Rate'] = data['Student Attendance Rate'].fillna(data['Student Attendance Rate'].dropna().mean())
data['Average ELA Proficiency'] = data['Average ELA Proficiency'].fillna(data['Average ELA Proficiency'].dropna().mean())
data['Average Math Proficiency'] = data['Average Math Proficiency'].fillna(data['Average Math Proficiency'].dropna().mean())




np.unique(data['Supportive Environment Rating'])
d = {'Exceeding Target': 3, 'Meeting Target': 2, 'Approaching Target': 1, 'Not Meeting Target': 1}



data['Supportive Environment Rating'] = data['Supportive Environment Rating'].map(d)
data['Rigorous Instruction Rating'] = data['Rigorous Instruction Rating'].map(d)
data['Collaborative Teachers Rating'] = data['Collaborative Teachers Rating'].map(d)
data['Effective School Leadership Rating'] = data['Effective School Leadership Rating'].map(d)
data['Strong Family-Community Ties Rating'] = data['Strong Family-Community Ties Rating'].map(d)
data['Trust Rating'] = data['Trust Rating'].map(d)

d2 = {'Exceeding Target': 3, 'Meeting Target': 2, 'Approaching Target': 1, 'Not Meeting Target': 1}
data['Student Achievement Rating'] = data['Student Achievement Rating'].map(d2)




x = list(data.columns)


def get_bins(no):
    if no == 0 :
        return 0
    elif no > 0 and no <= 15 :
        return 1
    elif no > 15 and no <= 30 :
        return 2
    elif no > 30 and no <= 45 :
        return 3
    elif no > 60 and no <= 75 :
        return 4
    else: 
        return 5



v_features = data.iloc[:,41:]

'''
v_features = v_features.drop(columns = ('Grade 3 ELA - All Students Tested','Grade 3 Math - All Students tested',
              'Grade 4 ELA - All Students Tested','Grade 4 Math - All Students Tested',
              'Grade 5 ELA - All Students Tested','Grade 5 Math - All Students Tested',
              'Grade 6 ELA - All Students Tested','Grade 6 Math - All Students Tested',
              'Grade 7 ELA - All Students Tested','Grade 7 Math - All Students Tested',
              'Grade 8 ELA - All Students Tested','Grade 8 Math - All Students Tested'))
'''


v_features = list(v_features.columns)
for i, cn in enumerate(data[v_features]):
    data[cn] = data[cn].apply(lambda x: get_bins(x))






def get_bin(no):
    if no == 0:
        return 0
    elif no > 0 and no <= 0.33 :
        return 1
    elif no > 0.33 and no <= 0.67 :
        return 2
    else:
        return 3


x_features = ['Percent Asian', 'Percent Black','Percent Hispanic','Percent White','Student Attendance Rate',
         'Percent of Students Chronically Absent','Percent ELL']

for i, cn in enumerate(data[x_features]):
    data[cn] = data[cn].apply(lambda x: get_bin(x))




data['Community School?'] = pd.get_dummies(data['Community School?'])







final = data[['Student Achievement Rating','School Name', 'SED Code','Economic Need Index', 'School Income Estimate', 
         'Community School?', 'Percent ELL','Percent Asian', 'Percent Black','Percent Hispanic','Percent White','Student Attendance Rate',
         'Percent of Students Chronically Absent','Rigorous Instruction Rating','Collaborative Teachers Rating',
         'Supportive Environment Rating','Effective School Leadership Rating','Strong Family-Community Ties Rating',
         'Trust Rating','Average ELA Proficiency','Average Math Proficiency','ELA Tests','Math Tests']]


'Grade 3 ELA - All Students Tested','Grade 3 Math - All Students Tested',
              'Grade 4 ELA - All Students Tested','Grade 4 Math - All Students Tested',
              'Grade 5 ELA - All Students Tested','Grade 5 Math - All Students Tested',
              'Grade 6 ELA - All Students Tested','Grade 6 Math - All Students Tested',
              'Grade 7 ELA - All Students Tested','Grade 7 Math - All Students Tested',
              'Grade 8 ELA - All Students Tested','Grade 8 Math - All Students Tested'




data1['ELA Tests'] = data1['Grade 3 ELA - All Students Tested'] + data1['Grade 4 ELA - All Students Tested'] + data1['Grade 5 ELA - All Students Tested'] + data1['Grade 6 ELA - All Students Tested'] + data1['Grade 7 ELA - All Students Tested'] + data1['Grade 8 ELA - All Students Tested']

data1['Math Tests'] = data1['Grade 3 Math - All Students tested'] + data1['Grade 4 Math - All Students Tested'] + data1['Grade 5 Math - All Students Tested'] + data1['Grade 6 Math - All Students Tested'] + data1['Grade 7 Math - All Students Tested'] + data1['Grade 8 Math - All Students Tested']



x = pd.qcut(data1['ELA Tests'], 5).value_counts()
y = pd.qcut(data1['Math Tests'], 5).value_counts()



data1.loc[(data1['ELA Tests'] >= 0) & (data1['ELA Tests'] <= 176), 'ELA Tests'] = 1
data1.loc[(data1['ELA Tests'] > 176) & (data1['ELA Tests'] <= 253), 'ELA Tests'] = 2
data1.loc[(data1['ELA Tests'] > 253) & (data1['ELA Tests'] <= 324), 'ELA Tests'] = 3
data1.loc[(data1['ELA Tests'] > 324) & (data1['ELA Tests'] <= 454), 'ELA Tests'] = 4
data1.loc[data1['ELA Tests'] > 454, 'ELA Tests'] = 5



data1.loc[(data1['Math Tests'] >= 0) & (data1['Math Tests'] <= 173), 'Math Tests'] = 1
data1.loc[(data1['Math Tests'] > 173) & (data1['Math Tests'] <= 251), 'Math Tests'] = 2
data1.loc[(data1['Math Tests'] > 251) & (data1['Math Tests'] <= 325), 'Math Tests'] = 3
data1.loc[(data1['Math Tests'] > 325) & (data1['Math Tests'] <= 454), 'Math Tests'] = 4
data1.loc[data1['Math Tests'] > 454, 'Math Tests'] = 5



data1 = data1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,146,147]]


col = list(data.columns)


data1 = data.loc[:, 'School Name':'Grade 8 Math 4s - Economically Disadvantaged']

######### Modeling



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


## Before Feature Engineering

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')




train, test = train_test_split(data1, test_size=0.25, random_state=50)



x_train = train.drop(columns = ('Student Achievement Rating'))
y_train = train[['Student Achievement Rating']]

x_test = test.drop(columns = ('Student Achievement Rating'))
y_test = test[['Student Achievement Rating']]



features = list(x_train.columns)
pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                      ('scaler', MinMaxScaler())])



train_set = pipeline.fit_transform(x_train)
test_set = pipeline.transform(x_test)



model = RandomForestClassifier(n_estimators=100, random_state=10, 
                               n_jobs = -1)
# 10 fold cross validation
cv_score = cross_val_score(model, train_set, y_train, cv = 10, scoring = scorer)





Score = round(cv_score.mean(), 4)



model.fit(train_set, y_train)

# Feature importances into a dataframe
feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importances.head()




###### Before Feature Engineering


scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')




train, test = train_test_split(final, test_size=0.25, random_state=50)





x_train = train.loc[:, 'Economic Need Index': 'Math Tests']
y_train = train[['Student Achievement Rating']]

x_test = test.loc[:, 'Economic Need Index': 'Math Tests']
y_test = test[['Student Achievement Rating']]

features = list(x_train.columns)
pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                      ('scaler', MinMaxScaler())])




train_set = pipeline.fit_transform(x_train)
test_set = pipeline.transform(x_test)



model = RandomForestClassifier(n_estimators=100, random_state=10, 
                               n_jobs = -1)
# 10 fold cross validation
cv_score = cross_val_score(model, train_set, y_train, cv = 10, scoring = scorer)





Score = round(cv_score.mean(), 4)






model.fit(train_set, y_train)

# Feature importances into a dataframe
feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importances.head()






# Model imports
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier


import warnings 
from sklearn.exceptions import ConvergenceWarning

# Filter out warnings from models
warnings.filterwarnings('ignore', category = ConvergenceWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Dataframe to hold results
model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])

def cv_model(train, train_labels, model, name, model_results=None):
    """Perform 10 fold cross validation of a model"""
    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer)
    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model': name, 
                                                           'cv_mean': cv_scores.mean(), 
                                                            'cv_std': cv_scores.std()},
                                                           index = [0]),
                                             ignore_index = True)

        return model_results




model_results = cv_model(train_set, y_train, LinearSVC(), 
                'LSVC', model_results)


model_results = cv_model(train_set, y_train, 
                         GaussianNB(), 'GNB', model_results)


model_results = cv_model(train_set, y_train, 
                          LinearDiscriminantAnalysis(), 
                          'LDA', model_results)


model_results = cv_model(train_set, y_train, 
                         RidgeClassifierCV(), 'RIDGE', model_results)


model_results = cv_model(train_set, y_train, 
                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),
                         'EXT', model_results)



model_results = cv_model(train_set, y_train,
                          RandomForestClassifier(100, random_state=10),
                              'RF', model_results)


















def plot_feature_importances(df, n = 20, threshold = None):
    plt.style.use('fivethirtyeight')
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    plt.rcParams['font.size'] = 12
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)
    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title('{} Most Important Features'.format(n),size = 18)
    plt.gca().invert_yaxis()
    if threshold:
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    return df






norm_fi = plot_feature_importances(feature_importances, threshold=0.95)






def kde_target(df, variable):
    """Plots the distribution of `variable` in `df` colored by the `Target` column"""
    
    colors = {1: 'red', 2: 'orange', 3: 'blue', 4: 'green'}

    plt.figure(figsize = (12, 8))
    
    df = df[df['Student Achievement Rating'].notnull()]
    
    for level in df['Student Achievement Rating'].unique():
        subset = df[df['Student Achievement Rating'] == level].copy()
        sns.kdeplot(subset[variable].dropna(), 
                    label = 'Poverty Level: {}'.format(level), 
                    color = colors[int(subset['Student Achievement Rating'].unique())])

    plt.xlabel(variable); plt.ylabel('Density');
    plt.title('{} Distribution'.format(variable.capitalize()));





kde_target(final, 'School Income Estimate')










######################### EDA

missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})
missing['percent'] = missing['total'] / len(data)





corr_matrix = data.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



corr_matrix.loc[corr_matrix['Grade 8 ELA - All Students Tested'].abs() > 0.9, corr_matrix['Grade 8 ELA - All Students Tested'].abs() > 0.9]



sns.heatmap(corr_matrix.loc[corr_matrix['Grade 8 ELA - All Students Tested'].abs() > 0.9, corr_matrix['Grade 8 ELA - All Students Tested'].abs() > 0.9],
            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


c = data.corr()['Student Achievement Rating'].reset_index()






corr_matrix = x_train.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]











##################### Plots





plt.figure(figsize = (15,8))
plt.hist(data['School Income Estimate'].dropna(), bins = 25)









plt.figure(figsize=(10,25))
gs = gridspec.GridSpec(ncols = 7, nrows = 2)
for i, cn in enumerate(data[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn], bins=5)
    ax.set_xlabel('')
    ax.set_title('Feature: ' + str(cn))
plt.show();





plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (15,8))
plt.bar(data['City'].value_counts().index, data['City'].value_counts().values)
plt.xticks(rotation = 'vertical')




pie_data = pd.DataFrame(data['Community School?'].value_counts().values,
                  index = data['Community School?'].value_counts().index, 
                  columns = [' '])

pie_data.plot(kind = 'pie', subplots = True, autopct = '%1.0f%%', figsize = (8,8))
plt.show();




data.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);



                  
                  
                  
                  
                  
                  
                  
                  from collections import OrderedDict
plt.figure(figsize = (20, 16))
plt.style.use('fivethirtyeight')



# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'Exceeding Target', 2: 'Meeting Target', 3: 'Approaching Target', 4: 'Not Meeting Target'})





# Create a list of buildings with more than 100 measurements
types = data.dropna(subset=['School Income Estimate'])
types = types['Student Achievement Rating'].value_counts()
types = list(types[types.values > 0].index)



for i in types:
    subset = data[data['Student Achievement Rating'] == i]
    sns.kdeplot(subset['School Income Estimate'].dropna(), label = i);





econ = data.dropna(subset=['Economic Need Index'])
econ = econ['Student Achievement Rating'].value_counts()
econ = list(econ[econ.values > 0].index)



for i in econ:
    subset = data[data['Student Achievement Rating'] == i]
    sns.kdeplot(subset['Economic Need Index'].dropna(), label = i);




hisp = data.dropna(subset=['Percent Hispanic'])
hisp = hisp['Student Achievement Rating'].value_counts()
hisp = list(hisp[hisp.values > 0].index)

for i in hisp:
    subset = data[data['Student Achievement Rating'] == i]
    sns.kdeplot(subset['Percent Hispanic'].dropna(), label = i);


attend = data.dropna(subset=['Student Attendance Rate'])
attend = attend['Student Achievement Rating'].value_counts()
attend = list(attend[attend.values > 0].index)

for i in attend:
    subset = data[data['Student Achievement Rating'] == i]
    sns.kdeplot(subset['Student Attendance Rate'].dropna(), label = i);





rig = data.dropna(subset=['Rigorous Instruction Rating'])
rig = rig['Student Achievement Rating'].value_counts()
rig = list(rig[rig.values > 0].index)

for i in rig:
    subset = data[data['Student Achievement Rating'] == i]
    sns.kdeplot(subset['Rigorous Instruction Rating'].dropna(), label = i);





achieve = data['Student Achievement Rating'].value_counts().sort_index()

achieve.plot.bar(figsize = (10,5), edgecolor = 'k')
plt.xlabel('Poverty Level');
plt.xticks(rotation = 0);

                  
                  
                  
                  
plt.figure(figsize = (15,8))
plt.hist(data['School Income Estimate'].dropna(), bins = 25)

