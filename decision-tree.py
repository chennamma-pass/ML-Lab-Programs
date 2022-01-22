#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


df=pd.read_csv('zoo.csv',index_col=False)


# In[9]:


df.head()


# In[10]:


class_type_output = df["class_type"]
df = df.drop("class_type", axis=1).drop("animal_name",axis=1)
print(df)


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, class_type_output, test_size=0.20)


# In[12]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)


# In[13]:


y_prediction = classifier.predict(x_test)
y_prediction


# In[14]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
confusion_matrix(y_test,y_prediction)


# In[15]:


print(classification_report(y_test, y_prediction))


# In[16]:


print(accuracy_score(y_test, y_prediction))


# In[17]:


predicted_class = list(y_prediction)
actual_class = list(y_test)
for i in range(len(predicted_class)):
    print("Predicted class =", predicted_class[i],"\tActual class =",actual_class[i])


# In[ ]:





# In[ ]:




