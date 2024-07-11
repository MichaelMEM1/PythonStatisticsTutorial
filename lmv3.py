#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python Version 3 

# ## BSGP 7030 Summer 2024

# ### Importing packages

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import numpy as np 

import sys

import os

from sklearn.linear_model import LinearRegression




# ### Making the python code executable on the first first argument 
# 

# In[2]:


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_dataframe.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]


# ### Extract the filename with and without extension

# In[3]:


csv_filename = os.path.basename(csv_file)
csv_title = os.path.splitext(csv_filename)[0]  # Get filename without extension


# ### Reading the csv file into a pandas dataframe

# In[4]:


df = pd.read_csv(csv_file)


# ### Plotting the data in the CSV file

# In[ ]:


plt.figure(figsize=(8, 6))  # specify the figure size
plt.scatter(df['x'], df['y'], label = "Data Points", marker='o', color='blue', alpha=0.5)  # Plotting x vs y
plt.title(f'Data from {csv_title}')  # Adding a title
plt.xlabel('x')  # Adding label to x-axis
plt.ylabel('y')  # Adding label to y-axis
plt.legend()
plt.grid(True)  #  Adding gridlines
plt.tight_layout()  #  Adjusting layout
plt.savefig('py_original.png') # Save the plot as png
plt.show()  # Display the plot


# ### Setting up the linear model and reshaping the data for a numpy array 

# In[ ]:


model = LinearRegression()
x = np.array(df['x']).reshape(-1, 1)
y = np.array(df['y'])
model.fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)


# ### Plotting the linear regression model

# In[ ]:


plt.figure(figsize=(8, 6))  #  specifying the figure size
plt.scatter(df['x'], df['y'], label = "Data Points", marker='o', color='blue', alpha=0.5)  # Plotting x vs y
plt.plot(df['x'],y_pred , label = "Fit")
plt.title('Linear Regression Model Python')  # Adding a title
plt.legend()     # Adding Legened
plt.text (12,10,f'$R^2={r_sq}$') #adding R squared
plt.xlabel('x')  # Adding label to x-axis
plt.ylabel('y')  # Adding label to y-axis
plt.grid(True)  #  adding gridlines
plt.tight_layout()  # adjusting layout
plt.savefig('py_LinearRegr.png') # Saving the plot as png
plt.show()  # Display the plot

