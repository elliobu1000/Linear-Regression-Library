import numpy as np
import matplotlib.pyplot as plt
import lib

# Data extraction : 
data = lib.read_data()
df = data.data_frame('Salary_Data.csv')
X = df['YearsExperience']
Y = df['Salary']



# Data Deployment :
model = lib.LR()
y = model.fit(X,Y,50,0.05)
model.plot(X,Y)