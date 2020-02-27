#!/usr/bin/env python
# coding: utf-8

# In[1]:


from metaphone import doublemetaphone
import textdistance


# In[5]:


kata1 = str(input("Please type a sentence: "))
kata2 = str(input("Please type a sentence: "))

a=textdistance.levenshtein.normalized_similarity(str(doublemetaphone(kata1)), str(doublemetaphone(kata2)))
b=textdistance.levenshtein.normalized_similarity(kata1, kata2)

if a==1 and b<1:
    score=a
else :
    score=b
    
print(score)


# In[ ]:




