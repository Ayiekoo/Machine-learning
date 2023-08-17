#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns


# In[3]:


iris = sns.load_dataset('iris')
iris.head(10)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
sns.pairplot(iris, hue='species', size=2)


# In[5]:


# we can use Scikit-Learn to extract features matrix and target arrays in the dataframe
X_iris = iris.drop('species', axis=1)
X_iris.shape ## We use pandas dataframe


# In[6]:


y_iris = iris['species']
y_iris.shape


# In[7]:


### supervised learning
## consider an example of a simple linear regression
# we shall fit a line of data (x and y)
import matplotlib.pyplot as plt
import numpy as np


# In[8]:


rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y); ## data is labelled as x and y


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


model = LinearRegression(fit_intercept=True) ## passing the code intantiates fitting the intercept
model


# In[11]:


## we can rearrange the data
X = x[:, np.newaxis] # reshapes 1D array
X.shape


# In[12]:


# we can fit the model
### this is  a model-dependent internal computation
model.fit(X, y)


# In[ ]:


# the model.coef_ and model.intercept_ funbctions represent the slope and 
# intercept of the simple linear fit.


# In[13]:


model.coef_


# In[14]:


model.intercept_


# In[18]:


## we can predict the labels of unknown data
## we can evaluate this data based on the data that was not part of the training dataset
## we use the predict() method
xfit = np.linspace(-1, 11)


# In[21]:


Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)


# In[22]:


plt.scatter(x, y)
plt.plot(xfit, yfit);


# In[24]:


from sklearn.cross_valudation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                               random_state=1)


# In[ ]:




