# -*- coding: utf-8 -*-
#ver.ki.voeux - ripki.py

import pandas as pd

data = {'Nama':['John','Anna','Peter','Linda','Fifi','Ripki','Ripki'], 'Usia':[28,35,60,98,17,17,19], 'Kota':['bekasi','bogor','bogor','bogor','bandung','depok','Bogor']}

df = pd.DataFrame(data)

display(df[['Nama','Kota']].head(2))

import matplotlib.pyplot as plt

student_data = pd.read_csv("student_data.csv")

temp = pd.DataFrame(student_data)

display(temp.head(100))

student_data.shape

sum(student_data.shape)

student_data.size

plt.plot(data['Nama'], data['Kota'], marker='o')
plt.title('Whatever')
plt.xlabel('Nama')
plt.ylabel('Kota')
plt.xticks(rotation=90)
plt.grid(True)

student_data.loc[5,["Group","Name","Height","Weight"]]

student_data.loc[5:9,["Group","Name","Height","Weight"]]

# student_data.groupby(['Group', 'Name']).count().reset_index()
student_data.groupby('Group').count().reset_index()

student_data[student_data["Group"] == "A"]["Height"].max()

student_data.groupby(['Group'])['Height'].max()
# student_data.groupby(['Group'])['Height'].min()
# max and min or whatever

temp.groupby(['Group'])['Height'].min().reset_index()

student_data.loc[10]

student_data.at[10, "Name"] = "Valentine Sanchez"

student_data.loc[10]

plt.plot(data['Nama'], data['Kota'], marker='o')
plt.title('Whatever')
plt.xlabel('Name')
plt.ylabel('Grup')
plt.xticks(rotation=90)
plt.grid(True)

student_data["Math_score"] = (student_data["Maths_Test_Score"] + student_data["Maths_Quiz_Score"] + student_data["Maths_Assignment_Score"])/3
student_data["Science_score"] = (student_data["Science_Test_Score"] + student_data["Science_Quiz_Score"] + student_data["Science_Assignment_Score"])/3
student_data["Literature_score"] = (student_data["Literature_Test_Score"] + student_data["Literature_Quiz_Score"] + student_data["Literature_Assignment_Score"])/3

student_data.head()

avg_score_data = student_data.groupby(['Group'])[['Math_score','Science_score','Literature_score']].mean().reset_index()

avg_score_data

avg_score_data.plot.bar(x='Group', color=['#FF667D', '#F0F0F0', '#F9F55F'])

avg_score_data = student_data.groupby(['Group'])['Height'].mean().reset_index()

avg_score_data.plot.bar(x ='Group', color=['#FF667D', '#F0F0F0', '#F9F55F'])

avg_score_data.plot.scatter(x="Math_score", y="Science_score", color="#F7FF6D")

student_data.dtypes

student_data.describe()

student_data['Science_score'].var()

student_data['Math_score'].plot.hist(bins=100, alpha=0.5)

student_data["Math_score"].corr(student_data["Science_score"])

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

diabeta = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

display(diabeta)

diabeta['Target'] = diabetes.target

display(diabeta)

import seaborn as sns

correlation = diabeta.corr()

sns.heatmap(correlation, annot=True)

diabeta.isna().sum()

student_data.isna().sum()

import json

json_student = student_data["Name"].to_json(orient='records', indent=4)

print(json_student)

diabeta['Target'] = diabetes.target

correlation = diabeta.corrwith(diabeta['Target'])

selected_features = correlation.abs().nlargest(5).index

print(selected_features)

