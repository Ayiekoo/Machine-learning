#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing requried packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[2]:


df = "C:/Users/Ayieko/Desktop/python/realtor-data.csv/realtor-data.csv"
df = pd.read_csv(df)
print(df)


# In[3]:


df.head(10)


# In[4]:


#####  create a scatter plot to see the relationship between any two variables, such as house size and price. This will help you determine if there is a correlation between the two variables. ####
import matplotlib.pyplot as plt

plt.scatter(df['house_size'], df['price'])
plt.xlabel('House size')
plt.ylabel('Price')
plt.show()


# In[5]:


###### a bar chart to visualize the count of houses in each city. This will help you understand which cities have the most listings#########

city_count = df.groupby('city').size().reset_index(name='count')
plt.bar(city_count['city'], city_count['count'])
plt.xticks(rotation=90)
plt.xlabel('City')
plt.ylabel('Count')
plt.show()


# In[6]:


###  a histogram to visualize the distribution of house prices. This will help you understand the range of prices and how they are distributed.  ###

plt.hist(df['price'], bins=10)
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()


# In[7]:


###### a heatmap to visualize the correlation between different variables####
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()


# In[ ]:




