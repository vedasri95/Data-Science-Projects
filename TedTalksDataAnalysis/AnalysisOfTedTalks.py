
# coding: utf-8

# In[132]:


import pandas as pa
import numpy as ny
from scipy import stats

data = pa.read_csv('ted_main.csv')
data.columns


# In[133]:


data = data[['name', 'title', 'url', 'description', 'main_speaker', 'speaker_occupation', 'num_speaker', 'event', 'duration', 'film_date', 'published_date', 'languages', 'comments', 'views', 'ratings', 'tags', 'related_talks']]


# In[134]:


data.columns


# In[135]:


del data['name']#deleted the attribute 'name' since it doesnot give any useful information.


# In[136]:


import datetime
data['film_date'] = data['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
data['duration'] = data['duration']/60


# In[137]:


data.head(5)


# In[138]:


data['published_date'] = data['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))


# In[139]:


data.head(5)


# In[140]:


##################### ANALYSING VIEWS OF TED TALKS ######################


# In[141]:


most_popular_talks = data[['title', 'views']].sort_values('views',ascending =False)[:10]
most_popular_talks
# top 10 most viewed ted talks


# In[143]:


sb.distplot(data['views'])


# In[144]:


sb.distplot(data[data['views'] < 0.5e7]['views'])


# In[145]:


data['views'].describe()


# In[146]:


#############SOME ANALYSIS OF COMMENTS IN TED TALKS ####################


# In[147]:



data['comments'].describe()


# In[148]:


plt.figure(figsize=(15,9))
sb.distplot(data['comments'])


# In[149]:


plt.figure(figsize=(15,9))
sb.distplot(data[data['comments'] < 1000]['comments'])


# In[150]:


#### CORRELATION BETWEEN VIEWS AND COMMENTS ######


# In[151]:


g = sb.pairplot(data, vars=["views", "comments"])


# In[152]:


data[['views','comments']].corr()


# In[153]:


##### 10 most commented talks #########


# In[154]:


most_comm_talks = data[['title', 'main_speaker', 'tags', 'views', 'comments']].sort_values('comments', ascending=False)[:20]
most_comm_talks


# In[155]:


##### above table, under tags section many contain 'culture'. highest comments first one since many people have different opinions on this


# In[156]:


#### ANALYSIS ON SPEAKERS OF TED TALKS #############


# In[157]:


speaker_data = data.groupby('main_speaker').count().reset_index()[['main_speaker', 'title']]


# In[158]:



speaker_data.columns = ['speaker name', 'appearances']
speaker_data = speaker_data.sort_values('appearances', ascending=False)
speaker_data.head(10)



# In[159]:


speaker_data['appearances'].describe()


# In[160]:


speaker_views_data = data.groupby('main_speaker')['views'].sum().reset_index()[['main_speaker','views']]


# In[161]:


speaker_views_data.columns = ['speaker name', 'Agg views']
speaker_views_data = speaker_views_data.sort_values('Agg views', ascending=False)
result = pa.merge(speaker_views_data, speaker_data, on='speaker name')


# In[162]:


result.head(10)


# In[163]:


result[['Agg views','appearances']].corr()


# In[164]:


### Ken robinson though he appeared only lessno of times, his videos have higher views than Hans Rosling who appeared highest number of times.


# In[165]:


occupation_data = data.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'views']]
occupation_data.columns = ['occupation', 'appearances']
occupation_data = occupation_data.sort_values('appearances', ascending=False)


# In[166]:


occupation_data.head(20)


# In[167]:


plt.figure(figsize=(15,10))
sb.barplot(x='occupation', y='appearances', data=occupation_data.head(20))
plt.show()


# In[168]:


speaker_comments = data.groupby('main_speaker')['comments'].sum().reset_index()[['main_speaker','comments']]
speaker_comments.columns = ['speaker name', 'Agg comments']
speaker_comments = speaker_comments.sort_values('Agg comments', ascending=False)
result2 = pa.merge(speaker_comments, speaker_data, on='speaker name')
result2.head(20)


# In[224]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 10))
sb.boxplot(x='speaker_occupation', y='views', data=data[data['speaker_occupation'].isin(occupation_data.head(25)['occupation'])], palette="muted", ax =ax)
ax.set_ylim([0, 0.4e7])
plt.show()


# In[ ]:






# In[170]:


####################### ANALYSIS ON DURATION OF TED TALKS ###########


# In[171]:



data['duration'].describe()


# In[172]:


data[data['duration'] == 87.6]


# In[173]:


sb.pairplot(data[data['duration'] < 30], vars=["views", "duration"])



# In[174]:


data[['views','duration']].corr()


# In[175]:


# Pearson coefficient is quite small. SO almost zero correlation between duration and views in ted talks. probably the viewers tend to watch longer talks also if the content is worth watching.


# In[176]:


################ ANALYSIS OF RATINGS ###################


# In[177]:


ratings = data['ratings']


# In[178]:



from collections import OrderedDict
from datetime import date


# In[179]:


import ast
data['ratings'] = data['ratings'].apply(lambda x: ast.literal_eval(x))


# In[180]:


data.head()


# In[181]:


data['funny'] = data['ratings'].apply(lambda x: x[0]['count'])




# In[182]:


data['ratings'][0]


# In[183]:


data['Confusing'] = data['ratings'].apply(lambda x: x[5]['count'])
data['Inspiring'] = data['ratings'].apply(lambda x: x[13]['count'])
data['Informative'] = data['ratings'].apply(lambda x: x[6]['count'])
data['Informative'][0]


# In[184]:


data.head()


# In[185]:


#top 10 most confusing talks


# In[186]:


data[['title', 'main_speaker', 'views', 'published_date', 'Confusing']].sort_values('Confusing', ascending=False)[:10]


# In[187]:


#top 10 most inspirational talks


# In[188]:


data[['title', 'main_speaker', 'views', 'published_date', 'Inspiring']].sort_values('Inspiring', ascending=False)[:10]


# In[189]:


#top 10 most funny talks


# In[190]:


data[['title', 'main_speaker', 'views', 'published_date', 'funny']].sort_values('funny', ascending=False)[:10]


# In[191]:


#top 10 most informative talks


# In[192]:


data[['title', 'main_speaker', 'views', 'published_date', 'Informative']].sort_values('Informative', ascending=False)[:10]


# In[193]:


#Analasying languages in ted talks


# In[194]:


data['languages'].describe()


# In[195]:


sb.pairplot(data, vars=['languages','views'])
plt.show()
data[['views','languages']].corr()


# In[196]:


## ANALYSING EVENTS


# In[197]:


events_df = data[['title', 'event']].groupby('event').count().reset_index()
events_df.columns = ['event', 'Num of talks']
events_df = events_df.sort_values('Num of talks', ascending=False)
events_df.head(10)


# In[198]:


events_df['Num of talks'].describe()


# In[199]:


event_views = data.groupby('event')['views'].sum().reset_index()[['event','views']]
event_views.columns = ['event', 'Aggregate views']
event_views = event_views.sort_values('Aggregate views', ascending=False)
result2 = pd.merge(event_views, events_df, on='event')
result2.head(10)


# In[200]:


# ANALYSING MONTH AND YEAR


# In[201]:


month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# In[202]:


data['month'] = data['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

month_df = pd.DataFrame(data['month'].value_counts()).reset_index()
month_df.columns = ['month', 'talks']


# In[203]:


month_df


# In[204]:


sb.barplot(x='month', y='talks', data=month_df, order=month_order)


# In[205]:


data['year'] = data['film_date'].apply(lambda x: x.split('-')[2])
year_df = pd.DataFrame(data['year'].value_counts().reset_index())
year_df.columns = ['year', 'talks']

plt.figure(figsize=(12,10))
sb.pointplot(x='year', y='talks', data=year_df)


# In[206]:


# ANALYSIS ON RELATED VIDEOS


# In[207]:


data['related_talks'] = data['related_talks'].apply(lambda x: ast.literal_eval(x))


# In[208]:


s = data.apply(lambda x: pd.Series(x['related_talks']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'related'
s.head(15)


# In[209]:


related_talk_data = data.drop('related_talks', axis=1).join(s)
related_talk_data['related'] = related_talk_data['related'].apply(lambda x: x['title'])
related_dict = dict(related_talk_data['title'].drop_duplicates())
related_dict = {v: k for k, v in related_dict.items()}
related_talk_data['title'] = related_talk_data['title'].apply(lambda x: related_dict[x])
related_talk_data['related'] = related_talk_data['related'].apply(lambda x: related_dict[x])
related_talk_data = related_talk_data[['title', 'related']]
graph_edges = list(zip(related_talk_data['title'], related_talk_data['related']))


# In[210]:


import networkx as network
graph = network.Graph()
graph.add_edges_from(graph_edges)
plt.figure(figsize=(20, 20))
network.draw(graph, node_color = 'b')


# In[211]:


my_dict = dict(graph.degree)
degree_data = pd.DataFrame(list(my_dict.items()),columns=['node','degree'])
degree_data.head(10)


# In[212]:


plt.figure(figsize=(10,9))
sb.distplot(degree_data['degree'])


# In[213]:


degree_data.describe()


# In[214]:


network.density(graph)


# In[215]:


## analysing ted themes


# In[216]:


import ast
data['tags'] = data['tags'].apply(lambda x: ast.literal_eval(x))


# In[217]:


theme_data = data.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
theme_data.name = 'tags'
theme_data = data.drop('tags', axis=1).join(theme_data)
theme_data.head()


# In[218]:


pop_themes = pd.DataFrame(theme_data['tags'].value_counts()).reset_index()
pop_themes.columns = ['talk theme', 'Num of talks']
pop_themes.head(10)


# In[219]:


plt.figure(figsize=(9,5))
sb.barplot(x='talk theme', y='Num of talks', data=pop_themes.head(10))
plt.show()

