#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[5]:


def randomization(n):
    A = np.random.random(size = n).reshape(n,1)
    return A


# In[6]:


def operations(h,w):
    A = np.random.random(size=(h,w))
    B = np.random.random(size =(h,w))
    S = A + B
    return A, B, S


# In[7]:


def Norm(A,B):
    S = A + B
    return np.linalg.norm(S)


# In[8]:


def neural_network(inputs,weights):
    Z = np.tanh(weights.T.dot(inputs))
    return Z
    


# In[9]:


def scalar_function(x,y):
    if(x>y):
        return x*y
    else:
        return x/y


# In[10]:


def vector_function(x,y):
    func = np.vectorize(scalar_function)
    return func(x,y)


# In[11]:


n = 5
randomization(n)


# In[14]:


h=3 
w = 3
operations(h,w)


# In[16]:


A = np.array([[1,2,3]]).T
B = np.array([[1,2,3]]).T
Norm(A,B)


# In[19]:


inputs = (2,5)
weights = np.array([[9, 2, 3], [3, 4, 5]])
neural_network(inputs,weights)


# In[21]:


x = 7
y =10
scalar_function(x,y)


# In[22]:


vector_function(x,y)


# In[ ]:




