#!/usr/bin/env python
# coding: utf-8

# In[1]:


atletas = ['alirio helio wagner fernando_mineiro fred duda marcellus julio_cesar daniel jefferson_william ian coloneze babby',
           'arthur estevam valtinho alex_garcia giovannoni rodrigo rossi nezinho marcio_cipriano mineiro',
           'arthur luiz_gustavo fabio alex_garcia giovannoni rafael rossi nezinho bruninho marcio_cipriano bruno alirio tischer',
           'arthur ronald fabio alex_garcia giovannoni rossi nezinho bruninho marcio_cipriano bruno alirio tischer',
           'marcelinho feliz shilton kojo benite chupeta duda marquinhos diego_marques caio_torres alexandre olivinha gege douglas zanotti',
           'marcelinho danielzinho shilton laprovittola benite chupeta marquinhos diego_marques leo washam olivinha alef gege douglas felicio stafleu meyinsse',
           'hermann marcelinho danielzinho laprovittola benite chupeta dede marquinhos diego leo olivinha alef gege felicio fernando mingau caique gigante meyinsse',
           'marcelinho rafa_luz gigante ramon marquinhos rafael_mineiro jp_batista danielzinho olivinha gege mingau robinson meyinsse',
           'gui_deodato shilton stefano valtinho alex_garcia jefferson henrique michael gege gui_santos leo_meindl gabriel_jau',
           'yago elinho jhonatan_luz lucas_dias victor_andre kyle_fuller alex du_sommer deryk dikembe david_nesbitt eddy guilherme_hubner',
           'balbi rossetto matheusinho joao_matheus deryk kevin marquinhos jhonathan_luz aieser olivinha nesbitt ruan_miranda varejao rafael_mineiro joao_vitor']

times = ['Flamengo', 'Brasilia','Brasilia','Brasilia','Flamengo','Flamengo','Flamengo','Flamengo','Bauru','Paulistano',
         'Flamengo']

tecnicos =['chupeta_coach','lula_ferreira','jose_vidal','jose_vidal','jose_neto','jose_neto','jose_neto','jose_neto',
           'demetrius','gustavo_conti','gustavo_conti']

temporadas = ['2008-09','2009-10','2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19']


# In[2]:


import pandas as pd
import numpy as np


# In[6]:


corpus_df =pd.DataFrame([temporadas, times, tecnicos, atletas]).T
corpus_df


# In[10]:


import nltk
import re


# In[11]:


# preprocessor that focuses on removing special characters, extra whitespace, digits, stopwords, and then lowercasing
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lowercas and remove special character\whitespace
    doc = re.sub('r[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


# In[12]:


# apply it to sample corpus
norm_corpus = normalize_corpus(atletas)
norm_corpus


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer


# In[14]:


# get bag of words features in sparse format
cv= CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix


# In[15]:


# view non-zero feature positions in the sparse matrix
print(cv_matrix)


# In[16]:


# view dense representation
# warning might give a memory error if data is too big
cv_matrix = cv_matrix.toarray()
cv_matrix


# In[17]:


# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
pd.DataFrame(cv_matrix, columns = vocab)


# In[18]:


# Constructing a co-occurrence matrix in python pandas ********************************************************************
# https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas
df = pd.DataFrame(cv_matrix, columns = vocab)
df_asint = df.astype(int)
coocc = df_asint.T.dot(df_asint)
coocc
# *************************************************************************************************************************


# In[19]:


# Reset diagonal
np.fill_diagonal(coocc.values, 0)
coocc


# In[20]:


import networkx as nx
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[35]:


G = nx.MultiGraph(coocc)


# In[36]:


nx.draw(G, node_color='c', node_size=900, with_labels=True)


# In[37]:


hst = nx.degree_histogram(G)
plt.hist(hst,bins=20,color='red')


# In[38]:


nx.draw(G, with_labels=True,node_color='y',node_size=800)


# In[41]:


import networkx as nx
from networkx.algorithms import bipartite


# In[42]:


B = nx.Graph(coocc)
nx.draw(B, with_labels = True, node_color = 'cyan', node_size = 750)


# In[43]:


social = nx.Graph(coocc)


# In[46]:


plt.figure(figsize = (12,7))
nx.draw_networkx(social, node_color = 'yellow', node_size = 700, with_labels = True)


# In[47]:


plt.figure(figsize = (12,7))
nx.draw(social,node_color='c',node_size=900, with_labels=True)


# In[59]:


sub = nx.subgraph(social,['duda','alirio','gege','deryk','rafael_mineiro'])
nx.draw(sub,node_color='c', node_size = 800, with_labels= True)


# In[64]:


# create a spring-layout of fb
pos = nx.spring_layout(social)

plt.figure(figsize = (10,7))

plt.axis('off')
nx.draw_networkx(social, pos=pos,with_labels=False, node_size=35)


# In[70]:


pairs = social.edges


# In[71]:


import operator
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def centrality_sort(centrality_dict):
    return sorted(centrality_dict.items(),key=operator.itemgetter(1))


# In[72]:


g = nx.Graph()
g.add_edges_from(pairs)

nx.draw(g, with_labels=True)


# In[73]:


#centrality : which nodes have the highest /lowest degree centrality

degree_cent = nx.degree_centrality(g)
degree_sorted = centrality_sort(degree_cent)

print('----------------------Degree Centrality--------------------------')
print('Highest degree:',degree_sorted[-5:])
print('\n')

print('Lowest degree:',degree_sorted[:5])


# In[74]:


#betweenness centrality : which nodes have the highest /lowest betweenness centrality

between_cent = nx.betweenness_centrality(g)
between_sorted = centrality_sort(between_cent)

print('----------------------Betwenness Centrality--------------------------')
print('Highest degree:',between_sorted[-5:])
print('\n')

print('Lowest degree:',between_sorted[:5])


# In[75]:


#closenness centrality : which nodes have the highest and lowest closeness centrality

closeness_cent = nx.closeness_centrality(g)
closeness_sorted = centrality_sort(closeness_cent)

print('----------------------Closenness Centrality--------------------------')
print('Highest degree:',closeness_sorted[-5:])
print('\n')

print('Lowest degree:',closeness_sorted[:5])


# In[76]:


highest_degree = [node[0] for node in degree_sorted[-20:]]


# In[77]:


#create a subgraph
plt.figure(figsize = (12,7))
sub = g.subgraph(highest_degree)
nx.draw(sub,with_labels=True, node_color = 'cyan', node_size = 900)


# In[78]:


highest_degree


# In[79]:


lowest_degree = [node[0] for node in degree_sorted[:20]]


# In[80]:


sub_low = g.subgraph(lowest_degree)
nx.draw(sub_low,with_labels=True)


# In[82]:


# create a digraph
d = nx.DiGraph()
plt.figure(figsize = (12,7))
d.add_edges_from(pairs)
nx.draw(d,with_labels=True)


# In[83]:


#get the in_degree_centrality , and out_degree_centrality

in_degree_centrality = nx.in_degree_centrality(d)
in_degree_sorted = centrality_sort(in_degree_centrality)

print('---------------------- in degree Centrality--------------------------')
print('Highest degree:',in_degree_sorted[-5:])
print('\n')

print('Lowest degree:',in_degree_sorted[:5])


#out_degree centrality
out_degree_centrality = nx.out_degree_centrality(d)
out_degree_sorted = centrality_sort(out_degree_centrality)

print('---------------------- out degree Centrality--------------------------')
print('Highest degree:',out_degree_sorted[-5:])
print('\n')

print('Lowest degree:',out_degree_sorted[:5])


# In[84]:


highest_in_degree = [node[0] for node in in_degree_sorted[-20:]]


# In[85]:


sub = d.subgraph(highest_in_degree)
nx.draw(sub,with_labels=True)


# In[86]:


lowest_in_degree = [node[0] for node in in_degree_sorted[:20]]
sub1 = d.subgraph(lowest_in_degree)
nx.draw(sub1,with_labels=True)


# In[87]:


highest_out_degree = [node[0] for node in out_degree_sorted[-20:]]
lowest_out_degree = [node[0] for node in out_degree_sorted[:20]]


# In[88]:


#plot highest out degree
plt.figure(figsize = (12,7))
sub = d.subgraph(highest_out_degree)
nx.draw(sub,with_labels=True)


# In[89]:


highest_out_degree


# In[90]:


#plot lowest out degree
sub12 = d.subgraph(lowest_out_degree)
nx.draw(sub12,with_labels=True)


# ## Delete specific edges{(A,B),(A,C)} from G

# In[140]:


tupla = [('Alírio','Chupeta'),('Hélio','Chupeta'),('Wagner','Chupeta'),('F.Mineiro','Chupeta'),('Fred','Chupeta'),
         ('Duda','Chupeta'),('Marcellus','Chupeta'),('J.Cesar','Chupeta'),('Daniel','Chupeta'),('J.William','Chupeta'),
         ('Ian','Chupeta'),('Coloneze','Chupeta'),('Babby','Chupeta'),('Arthur','Lula'),('Estevam','Lula'),
         ('Valtinho','Lula'),('A.Garcia','Lula'),('Giovannoni','Lula'),('Rodrigo','Lula'),('Rossi','Lula'),
         ('Nezinho','Lula'),('M.Cipriano','Lula'),('Mineiro','Lula'),('Arthur','J.Vidal'),('L.Gustavo','J.Vidal'),
         ('Fabio','J.Vidal'),('A.Garcia','J.Vidal'),('Giovannoni','J.Vidal'),('Rafael','J.Vidal'),('Rossi','J.Vidal'),
         ('Nezinho','J.Vidal'),('Bruninho','J.Vidal'),('M.Cipriano','J.Vidal'),('Bruno','J.Vidal'),('Alírio','J.Vidal'),
         ('Tischer','J.Vidal'),('Arthur','J.Vidal'),('Ronald','J.Vidal'),('Fabio','J.Vidal'),('A.Garcia','J.Vidal'),
         ('Giovannoni','J.Vidal'),('Rossi','J.Vidal'),('Nezinho','J.Vidal'),('Bruninho','J.Vidal'),
         ('M.Cipriano','J.Vidal'),('Bruno','J.Vidal'),('Alírio','J.Vidal'),('Tischer','J.Vidal'),('Marcelinho','J.Neto'),
         ('Feliz','J.Neto'),('Shilton','J.Neto'),('Kojo','J.Neto'),('Benite','J.Neto'),('Chupeta','J.Neto'),
         ('Duda','J.Neto'),('Marquinhos','J.Neto'),('D.Marques','J.Neto'),('C.Torres','J.Neto'),('Alexandre','J.Neto'),
         ('Olivinha','J.Neto'),('Gege','J.Neto'),('Douglas','J.Neto'),('Zanotti','J.Neto'),('Marcelinho','J.Neto'),
         ('Danielzinho','J.Neto'),('Shilton','J.Neto'),('Laprovittola','J.Neto'),('Benite','J.Neto'),
         ('Chupeta','J.Neto'),('Marquinhos','J.Neto'),('D.Marques','J.Neto'),('Leo','J.Neto'),('Washam','J.Neto'),
         ('Olivinha','J.Neto'),('Alef','J.Neto'),('Gege','J.Neto'),('Douglas','J.Neto'),('Felicio','J.Neto'),
         ('Stafleu','J.Neto'),('Meyinsse','J.Neto'),('Hermann','J.Neto'),('Marcelinho','J.Neto'),('Danielzinho','J.Neto'),
         ('Laprovittola','J.Neto'),('Benite','J.Neto'),('Chupeta','J.Neto'),('Dede','J.Neto'),('Marquinhos','J.Neto'),
         ('Diego','J.Neto'),('Leo','J.Neto'),('Olivinha','J.Neto'),('Alef','J.Neto'),('Gege','J.Neto'),
         ('Felicio','J.Neto'),('Fernando','J.Neto'),('Mingau','J.Neto'),('Caique','J.Neto'),('Gigante','J.Neto'),
         ('Meyinsse','J.Neto'),('Marcelinho','J.Neto'),('R.Luz','J.Neto'),('Gigante','J.Neto'),('Ramon','J.Neto'),
         ('Marquinhos','J.Neto'),('R.Mineira','J.Neto'),('JP.Batista','J.Neto'),('Danielzinho','J.Neto'),
         ('Olivinha','J.Neto'),('Gege','J.Neto'),('Mingau','J.Neto'),('Robinson','J.Neto'),('Meyinsse','J.Neto'),
         ('G.Deodato','Demétrius'),('Shilton','Demétrius'),('Stefano','Demétrius'),('Valtinho','Demétrius'),
         ('A.Garcia','Demétrius'),('Jefferson','Demétrius'),('Henrique','Demétrius'),('Michael','Demétrius'),
         ('Gege','Demétrius'),('G.Santos','Demétrius'),('L.Meindl','Demétrius'),('G.Jau','Demétrius'),
         ('Meyinsse','Demétrius'),('Yago','G.Conti'),('Elinho','G.Conti'),('J.Luz','G.Conti'),('L.Dias','G.Conti'),
         ('V.Andre','G.Conti'),('K.Fuller','G.Conti'),('Alex','G.Conti'),('D.Sommer','G.Conti'),('Deryk','G.Conti'),
         ('Dikembe','G.Conti'),('D.Nesbitt','G.Conti'),('Eddy','G.Conti'),('G.Hubner','G.Conti'),('Balbi','G.Conti'),
         ('Rossetto','G.Conti'),('Matheusinho','G.Conti'),('J.Matheus','G.Conti'),('Deryk','G.Conti'),
         ('Kevin','G.Conti'),('Marquinhos','G.Conti'),('J.Luz','G.Conti'),('Aieser','G.Conti'),('Olivinha','G.Conti'),
         ('Nesbitt','G.Conti'),('R.Miranda','G.Conti'),('Varejao','G.Conti'),('R.Mineiro','G.Conti'),('J.Vitor','G.Conti')]


# In[168]:


# define our function called simple_graph()
def simple_graph():
    #create our empty graph
    G = nx.Graph(tupla)
    
    #draw the graph
    nx.draw(G, with_labels=True,node_color='c',node_size=900)
    
    #define the nodelist
    nodelist = ['G.Conti','J.Neto','Demétrius','Lula','J.Vidal']


# In[171]:


# call simple_graph function
plt.figure(figsize = (12,8))
simple_graph()


# In[ ]:




