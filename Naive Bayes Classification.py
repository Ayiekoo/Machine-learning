#!/usr/bin/env python
# coding: utf-8

# <b><h1>Naive Bayes Classification </b></h1>
# 
# This is an algorithm used in supervised and unsupervised learning.
# The algorithm is suitable for high-dimensional datasets.
# The algorithm is:
#     
#  - Fast
#  - Has few tunable parameters

# <h2><b>Bayesian Classification </h2></b>
# 
# Naive Bayes classifiers are build on the Bayesian Classification methods.
# They rely in probabilities of the statistical quantities.
# For instance, if we want to find the probability of a label in a given dataset, 
# we can write it as <b>P(L | features)</b>.
# The Bayes' theorem indicates that we shoyuld express these probabilities as quantities to enable direct computation, as given by the formula:
# 
# <b><i>P</i>(L | features) = <i>P</i>(features | <i>L</i>)<i>P(L)</i> / P(features)</b>

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# <h1><b>Gaussian Naive Bayes</h1></b>
# 
# This classifier is based on the following assumption:
# 
#    <i><b>It is assumed that data from each label is drawn from a simple Gaussian distribution.</i></b>

# In[4]:


from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

# This plots data obtained from the Gaussian naive Bayes


# We can assume that the data is distributed by the Gaussian distribution without covariance between the dimensions.
# This is a very fast way of creating a simple model.
# We can fit this model by:
#     
#  - Finding the standard deviation of the points within each label.
#  - Finding the mran of the points within each label.
# 
# There's a slightly curved boundary in this classification; the boundary in Gaussian naive Bayes quadratic.
# 
# This Bayesian formalism enables probablistic classification, computed by the method; <b><i>predict_prob</b></i> method.

# In[ ]:


# We can look at another example of the Naive Bayes Classification


# <h2><b>Multinomial Naive Bayes</h2></b>
# 
# Assumption: It is assumed that the features are generated from simple multinomial distribution.
# The multinomial distribution defines the probabilities of observing counts among number categories.
# Therefore, the multinomial naive Bayes is appropriate for features presenting counts or count rates.
# 
# This algorithm uses the best-fit multinomial distribution model!
# 
# <b>Example: Text classification</b>
# 
# We shall classify test.
# Multinomial naive Bayes is usually used to classify tests.
# Here, features are related to word counts or frequencies to be classified. 

# In[7]:


from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names


# In[8]:


categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 
              'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5])


# To use data for machine learning, we must convert the content of each string into vector of numbers.
# In this approach, we use the <b>TF-IDF vectorizer</b> to create a pipeline that attaches it to a multinomial naive Bayes classifier:
#     

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# In[11]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# This pipeline applies the model to the training data, predicts labels for the test data.


# In[12]:


model.fit(train.data, train.target)
labels = model.predict(test.data)

# This model predicts test data.
# Now, the data is ready for evaluation to learn the performance of the estimator.
# For instance, the confusion matrix between the true and predicted labels is indicated.


# In[14]:


from sklearn.metrics import confusion_matrix
mat =  confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')

# This prints the confusion matrix for the multinomial naive Bayes classifier


# The Naive Bayes is based on stringent assumptions.
# Therefore, itr does not perform as complicated models!
# 
# It has the following advantages:
#  - It is extremely fast fir prediction and training
#  - It provides a straightforward probablistic prediction
#  - Easy interpretation
#  - Few tunable parameters
#  
# These advantages imply that the Naive Bayes classification ideal for initial baseline classification.
# The Naive Bayes classification works best under the following situations:
#  - When the assumptions match the data.
#  - With well-separated categories; when model complexity is vital.
#  - With high-dimensional data or when the complexity is not important.
#  
# <b>NOTE</b>: Well-separated data (model complexity) and high dimensional data (the importance of complexity in the project) are related in this way;
# The increasing dimensions of the dataset is much less likely for any two points to be found together. After all, the two points must be close in <b><i>every single dimesion</b></i> to be close overall.
# This suggests that clusters in high dimensions are separated, averagely, than clusters in low dimensions (assuming that the new dimensions actually add information).
# Therefore, simplistic classifiers like the Naive Bayes tend to work better than complicated classifiers as the dimensionality grows.
# When having enough data, even a simple model becomes very powerful!

# <h1><b>We can look at another example of Naive Bayes using the breast cancer dataset</h1></b>

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# Imports necessary libraries into the environment


# In[3]:


# Let's import the data into the environment
data = pd.read_csv("C:/Users/Ayieko/Desktop/breast cancer.csv")
print(data)


# In[4]:


data.head(10)


# We can generate a basic histogram of the diagnosis information in the data

# In[5]:


data["diagnosis"].hist()


# In[7]:


corr = data.iloc[:,:-1].corr(method="pearson")
cmap = sns.diverging_palette(250,354,80,60,center='dark',as_cmap=True)
sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidth=.2)


# In[9]:


data = data[["radius_mean", "texture_mean", "smoothness_mean", "diagnosis"]]
data.head(10)


# In[ ]:




