"""
@author: Yu Zhu, Christina Eng

The important dataset to get the conclusion of hightlight for each of the restaurant:
    TF-IDFList (tfidf_words) every restaurant's top topics score record to augment LDA analysis, 
    cluster_word, cluster_name then we can know competitors and words they have in common
    getLDAtopic --> LDAlist, a more precise way to get every restaurant's top topics
"""

from __future__ import print_function
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np

# ------------------------------------Text Data Preparation------------------------------------ #

# --------------------------------------------------------------------------- #
# Eliminaate all the obviously negative sentences 

def loadLexicon(fname):
    
    newLex = set()
    lex_conn = open(fname, encoding="utf8")
    
    # Add every word in the file to the set
    for line in lex_conn:
        newLex.add(line.strip()) # Remember to strip to remove the lin-change character
    lex_conn.close()
    return newLex


def processfun(review):
    
    negLex = loadLexicon('negative-words.txt')
    addString = ''
    
    # Split into sentences
    review = review.split('.')
    
    for sentence in review:
        sentence = sentence.lower()
        sentence = re.sub('[^a-z]', ' ', sentence)
        words = sentence.strip().split()

        # Check if the sentence has a negative word
        hasNeg = False
        wordpool = set()
        for word in words:
            wordpool.add(word)
            if bool(wordpool & negLex):
                hasNeg = True
                break
        if hasNeg:
            continue
        
        # Collect sentences that we need
        sentence_with_space = ' ' + sentence
        addString += sentence_with_space
        addString = addString.strip()
        
    return addString


# x is a dictionary
def restaurant_review_sifter(x):

    # Get a list of restaurants from the given dictionary
    restaurant_list = []
    for k in x.keys():
        restaurant_list.append(k)
    
    # i reps the index of restaurant
    processed_dict = {}
    for i in restaurant_list:
        reviewList = x[i]
        processed_reviewList = []
        for j in range(len(reviewList)):
            processed_review = processfun(reviewList[j])
            processed_review = " ".join(processed_review.split())
            processed_reviewList.append(processed_review)
            theList = list(filter(None, processed_reviewList))
            
        # Create the processed dictionary
        processed_dict[i] = theList
    
    return processed_dict
        
processed_chinese = restaurant_review_sifter(chinese_dict)
processed_mexican = restaurant_review_sifter(mexican_dict)
processed_italian = restaurant_review_sifter(italian_dict)


# ------------------------------------Conduct an analysis on finding competitors------------------------------------ #
# I analyzed different cusines separately
# Chinese cusine/ Mixican/ Italian

# Get a list of stopwords
stopword = stopwords.words('english')

# (By hand)
# Name lists
name = [] # Restaurant names and this will be used everywhere throughout this script
for i in processed_chinese.keys(): 
    name.append(i)

def getWords(processed_dict):
    
    # Two primary lists
    name = [] # Restaurant names
    synopses = [] # Reviews
    
    for i in processed_dict.keys(): 
        name.append(i)    
    for j in name:
        longString = ''.join(processed_dict[j])
        synopses.append(longString)
    
    clean_words_of_bag = []
    for i in synopses:
        longString = i
        stringList = longString.split()
        clean_synopses = []
        for j in stringList:
            if j in stopword:
                continue
            else:
                clean_synopses.append(j)
                longString = ' '.join(clean_synopses)
        clean_words_of_bag.append(longString)
        
    return clean_words_of_bag

clean_words_of_bag = getWords(processed_chinese)

# Italian reviews
# clean_words_of_bag = getWords(processed_italian)

# Mexican reviews
# clean_words_of_bag = getWords(processed_mexican)

'''
name = [] # Restaurant names
synopses = [] # Reviews

for i in processed_chinese.keys(): 
    name.append(i)

for i in processed_italian.keys(): 
    name.append(i)

for i in processed_mexican.keys(): 
    name.append(i)

for j in name:
    longString = ''.join(processed_chinese[j])
    synopses.append(longString)

for j in name:
    longString = ''.join(processed_italian[j])
    synopses.append(longString)
    
for j in name:
    longString = ''.join(processed_mexican[j])
    synopses.append(longString)

# Clean the stopwords
clean_words_of_bag = []
for i in synopses:
    longString = i
    stringList = longString.split()
    clean_synopses = []
    for j in stringList:
        if j in stopword:
            continue
        else:
            clean_synopses.append(j)
            longString = ' '.join(clean_synopses)
    clean_words_of_bag.append(longString)
'''

# --------------------------------------------------------------------------- #
# Tf-idf
# Conduct a feature extraction to get numbers
# Convert the clean_words_of_bag list into a tf-idf matrix
# Feed the collection of text documents directly to TfidfVectorizer and 
# go straight to the TF/IDF representation
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=None, ngram_range=(1,3))

# 54x2726 matrix - one row per restaurant, one column per phrase
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_words_of_bag)
terms = tfidf_vectorizer.get_feature_names()
dense = tfidf_matrix.todense()
len(dense[0].tolist()[0]) # the size of one row of the matrix contains the TF-IDF score for every phrase in our corpus for the first restaurant
                          # Some of them won't happen in the real reviews, we need to filter those out

# Calculate TF-IDF value for one restaurant
def getTF_IDFvalue(i):
    one_restaurant = dense[i].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(one_restaurant)), one_restaurant) if pair[1] > 0]
    #terms = tfidf_vectorizer.get_feature_names()
    listofValue = []
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1]*-1)
    for phrase, score in [(terms[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
        listofValue.append(str('{0: <20} {1}'.format(phrase, score)))
    
    return listofValue

# Get all the restaurants' TF-IDF scores and its distribution
TF_IDFList = [] # The most frequently occurred word in every restaurant's review
for i in range(0, len(clean_words_of_bag)):
    TF_IDFList.append(getTF_IDFvalue(i))


# --------------------------------------------------------------------------- #
# K-means clustering and find competitors
km = KMeans(n_clusters=5, max_iter=500, random_state=3425)

km.fit(tfidf_matrix)

# Save the function
joblib.dump(km, 'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')

clusters = km.labels_.tolist()

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

#  A dataframe put name and cluster number together   
restaurants = {'name':name, 'synopsis': clean_words_of_bag, 'cluster': clusters}
frame = pd.DataFrame(restaurants, index = clusters, columns = ['name', 'cluster'])
frame['cluster'].value_counts() # Number of restaurants per cluster (clusters from 0 to 4)


# Get one list for the group of competitors and the other one group the words where all the restaurants cluster around
cluster_word = [] # Items that are common among competitors
cluster_name = [] # Groups of competitors
for i in range(5):
    oneList = []
    for ind in order_centroids[i, :6]:
        oneList.append(terms[ind])
    cluster_word.append(oneList)
    
    twoList = []
    for n in frame.loc[i]['name'].values.tolist():
        twoList.append(n)
    cluster_name.append(twoList)


# ------------------------------------Latent Dirichlet Allocation Analysis------------------------------------ #
# The function to apply LDA analysis to get the top topics for each of the restaurant
def getLDAtopic(processed_dict, index):
    topic_num=3

    #tokenization
    tf_vectorizer = CountVectorizer(max_df=0.8, min_df=0.2, stop_words='english')
                                
    #read the dataset                
    docs = processed_dict[list(processed_dict.keys())[index]] # 1st restaurant, 2nd restaurant....

    #transform the docs into a count matrix
    matrix = tf_vectorizer.fit_transform(docs)

    #get the vocabulary
    vocab=tf_vectorizer.get_feature_names()
    
    #initialize the LDA model
    model = lda.LDA(n_topics=topic_num, n_iter=500)

    #fit the model to the dataset
    model.fit(matrix)

    #write the top terms for each topic
    top_words_num=3
    topic_mixes= model.topic_word_
    
    LDAtopic = []
    for i in range(topic_num): #for each topic
        sorted_indexes=np.argsort(topic_mixes[i])[len(topic_mixes[i])-top_words_num:]#get the indexes of the top-k terms in this topic
        sorted_indexes=sorted_indexes[::-1]#reverse to get the best first    
        for ind in sorted_indexes:
            LDAtopic.append(vocab[ind])
            
    return LDAtopic

# (By hand)
LDAlist = []
for i in range(len(list(processed_chinese.keys()))): # All the restaurants
    words = getLDAtopic(processed_chinese, i)
    LDAlist.append(words)

# Other LDAlist
'''
LDAlist = []
for i in range(len(list(processed_italian.keys()))): # All the restaurants
    words = getLDAtopic(processed_italian, i)
    LDAlist.append(words)

LDAlist = []
for i in range(len(list(processed_mexican.keys()))): # All the restaurants
    words = getLDAtopic(processed_mexican, i)
    LDAlist.append(words)
'''    


# ------------------------------------Find the Highlights------------------------------------ #
# TF-IDF scores based result for top 5 words for each of the restaurant
# Get the string value of the top 3 words list from the previous TF-IDF list
tfidf_words = []
for i in range(len(TF_IDFList)):
    wordsList = []
    for j in TF_IDFList[i][:3]:
        wordsList.append(j.split()[0])
    tfidf_words.append(wordsList)

# The function to get the highlight     
def getHighlight(rest_name):
    for i in range(len(cluster_name)):
        if rest_name in cluster_name[0]:
            index = name.index(rest_name)
            unique_maybe = set(LDAlist[index]) | set(tfidf_words[index])
            unique = unique_maybe - set(cluster_word[0])
        elif rest_name in cluster_name[1]:
            index = name.index(rest_name)
            unique_maybe = set(LDAlist[index]) | set(tfidf_words[index])
            unique = unique_maybe - set(cluster_word[1])
        elif rest_name in cluster_name[2]:
            index = name.index(rest_name)
            unique_maybe = set(LDAlist[index]) | set(tfidf_words[index])
            unique = unique_maybe - set(cluster_word[2])
        elif rest_name in cluster_name[3]:
            index = name.index(rest_name)
            unique_maybe = set(LDAlist[index]) | set(tfidf_words[index])
            unique = unique_maybe - set(cluster_word[3])
        elif rest_name in cluster_name[4]:
            index = name.index(rest_name)
            unique_maybe = set(LDAlist[index]) | set(tfidf_words[index])
            unique = unique_maybe - set(cluster_word[4])
    
    return unique

HighLight_bag_of_words = getHighlight('Big China')

# Chinese example
'''
getHighlight('Big China')
Out[33]: 
{'best',
 'big',
 'chicken',
 'chinese',
 'food',
 'order',
 'place',
 'shrimp',
 'wing'}
'''

# Italian example
#getHighlight('A Litteri')
#Out[17]: {'great', 'italian', 'litteri', 'selection', 'sub'}
'''
getHighlight('Al Volo')
Out[10]: 
{'carbonara',
 'food',
 'fresh',
 'good',
 'great',
 'lamb',
 'pasta',
 'place',
 'restaurant'}
''' 

# Mexican example
# getHighlight('Alero Restaurant')
# {'chicken', 'customer', 'food', 'good', 'just', 'mexican', 'place', 'service'}
# getHighlight('Toro Toro')
# {'food', 'place', 'restaurant', 'shrimp', 'tapas', 'toro'}

# A loop gives you an idea about all the restaurants' highlights
for i in range(len(name)):
    print(getHighlight(name[i]))

# ------------------------------------K-means Clustering Plot------------------------------------ #
# Multidimensional scaling
# For later plot purpose
dist = 1 - cosine_similarity(tfidf_matrix)

# Convert two components as we're plotting points in a two-dimensional plane
# Specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


# --------------------------------------------------------------------------- #
# Visualizing document clusters

# Set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

# (By hand)
# Chinese 

# Set up cluster names using a dict
cluster_names = {0: 'dishes, spicy, tofu, authentic', 
                 1: 'delivery, general tso', 
                 2: 'noodle, soup, pork, dumplings', 
                 3: 'wings, mumbo sauce, fries', 
                 4: 'egg rolls, dragon, huge'}

# Italian
'''
cluster_names = {0: 'ravioli, calamari, gnocchi, veal', 
                 1: 'brunch, happy hour, mimosas, osteria', 
                 2: 'lobster, tiramisu, lamb, reservation', 
                 3: 'deli, gelato, cafe', 
                 4: 'pizza, sandwhich, margherita'}
'''
# Mexican
'''
# Set up cluster names using a dict
cluster_names = {0: 'taco, bar, margaritas', 
                 1: 'cheese, quesadilla, beans, rice', 
                 2: 'burrito, lengua, pastor authentic', 
                 3: 'fajitas,, patio, enchiladas', 
                 4: 'brunch, dinner, happy hour'}
'''

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=name)) 

# Group by cluster
groups = df.groupby('label')

# Set up the plot
fig, ax = plt.subplots(figsize=(17, 9)) # Set the size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

# Iterate through groups to layer the plot
# Use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for n, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[n], color=cluster_colors[n], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # Changes apply to the x-axis
        which='both',      # Both major and minor ticks are affected
        bottom='off',      # Ticks along the bottom edge are off
        top='off',         # Ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # Changes apply to the y-axis
        which='both',      # Both major and minor ticks are affected
        left='off',      # Ticks along the bottom edge are off
        top='off',         # Ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  # sShow legend with only 1 point

# Add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)  
    
plt.show() # Show the plot

# --------------------------------------------------------------------------- #
# Hierarchical Clustering Application (As a supplement analysis for K-means clustering)
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) # Define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # Set the size
ax = dendrogram(linkage_matrix, orientation="right", labels=name);

plt.tick_params(\
    axis= 'x',          # Changes apply to the x-axis
    which='both',      # Both major and minor ticks are affected
    bottom='off',      # Ticks along the bottom edge are off
    top='off',         # Ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() # Show plot with tight layout

