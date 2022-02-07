#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("C:/Users/amans/Downloads/DSC Task 1 - Absenteeism_at_work.csv")


# In[4]:


data.head()


# In[6]:


data.columns


# In[7]:


data[data.columns].isnull().sum()


# In[8]:


data.shape


# In[9]:


data.describe()


# In[11]:


data['Reason for absence'].value_counts()


# In[67]:


data["Absenteeism time in hours"].describe()


# In[68]:


data["Absenteeism time in hours"].value_counts()


# In[ ]:


# It clearly looks like 7,104,48,112,56,120,80,64,32,5,40 could be outliers as less than <SD and data can be cleaned.
# But due to lack of data and lots of features, I would continue keeping the data constant as it wont have adverse affects on
# pattern we are going to see between between the features and abseentism.


# In[12]:


# Creating Scatter Plots and Bar Plots ( to understand the data and the patterns)


# In[14]:


data.plot.scatter(x="ID", y="Absenteeism time in hours")


# In[84]:


data.groupby(['ID']).count()


# In[82]:


data['ID'].value_counts()


# In[66]:


plt.bar(x=data["ID"], height=data["Absenteeism time in hours"],width=0.8)


# In[72]:


# Here we can see there is hardly any correlation of id with the absenteeism as the data is scattered along.
# Even though a case can be made ppl with id=3 have max hours of abseentism but this feature doesnt make that sense as sd is pretty low.
# Having said that id=28 ppl should be investigated.. also id=11,22


# In[77]:


data.groupby(['Reason for absence']).count()


# In[81]:


data['Reason for absence'].value_counts()


# In[78]:


data.plot.scatter(x="Reason for absence", y="Absenteeism time in hours")


# In[79]:


plt.bar(x=data["Reason for absence"], height=data["Absenteeism time in hours"],width=0.8)


# In[202]:


# disease number 13,19 needs awareness as for the rest leaving couple of outliers and some lack of data its pretty normal working hours


# In[203]:


# also 14 -> Diseases of the musculoskeletal system and connective tissue,Diseases of the genitourinary system,njury, poisoning and certain other consequences of external causes


# In[196]:


data.columns


# In[197]:


data['Day of the week'].value_counts()


# In[198]:


data.plot.scatter(x="Day of the week", y="Absenteeism time in hours")


# In[199]:


# this shows majority of ppl take of on monday and tueday( wednesday a bit less and so on decreasing)
# which makes sense


# In[88]:


data['Month of absence'].value_counts()


# In[85]:


data.groupby(['Month of absence']).count()


# In[ ]:


data['']


# In[86]:


data.plot.scatter(x="Month of absence", y="Absenteeism time in hours")


# In[97]:


plt.bar(x=data["Month of absence"], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# Nothing really solid can be deciphered here, where data is consistent( except month 0 which is nothing  and a peak in march but not high enough to get somewhere)


# In[98]:


data.groupby(['Seasons']).count()


# In[99]:


data["Seasons"].value_counts()


# In[100]:


data.plot.scatter(x="Seasons", y="Absenteeism time in hours")


# In[101]:


plt.bar(x=data["Seasons"], height=data["Absenteeism time in hours"],width=0.8)


# In[102]:


# Again seasons doesnt get us any pattern as its pretty consistent among the 4 seasons


# In[104]:


data.groupby(['Transportation expense']).count()


# In[105]:


data["Transportation expense"].value_counts()


# In[106]:


data.plot.scatter(x="Transportation expense", y="Absenteeism time in hours")


# In[107]:


plt.bar(x=data["Transportation expense"], height=data["Absenteeism time in hours"],width=0.8)


# In[108]:


# interesting trend here: rather than expensive transport and more absenntess, here its kinda opposite.
# again some consistent clusters in 179, 1118 especiallu but nothing concrete can be suggested here


# In[109]:


data.groupby(['Distance from Residence to Work']).count()


# In[110]:


data["Distance from Residence to Work"].value_counts()


# In[111]:


data.plot.scatter(x="Distance from Residence to Work", y="Absenteeism time in hours")


# In[112]:


plt.bar(x=data["Distance from Residence to Work"], height=data["Absenteeism time in hours"],width=0.8)


# In[113]:


# again nothing concrete can be established


# In[114]:


data.groupby(['Service time']).count()


# In[115]:


data["Service time"].value_counts()


# In[116]:


data.plot.scatter(x="Service time", y="Absenteeism time in hours")


# In[117]:


plt.bar(x=data["Service time"], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# between 10 and 18 the ansenetism , so keeping the service below or above could be tested for some of the employees,


# In[118]:


data.groupby(['Age']).count()


# In[119]:


data["Age"].value_counts()


# In[120]:


data.plot.scatter(x="Age", y="Absenteeism time in hours")


# In[121]:


plt.bar(x=data["Age"], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# this data just tells us out employyes who have just joined( or a bit young) and old emplyees( having family ) tend to
# to be less absent and follow their work. While in the age clusters of mid thirties we can see fluctuations but nothing too concrete 
# take a decison


# In[122]:


data.groupby(['Work load Average/day ']).count()


# In[127]:


# data["Work load Average/day" ].value_counts()


# In[125]:


data.plot.scatter(x="Work load Average/day ", y="Absenteeism time in hours")


# In[126]:


plt.bar(x=data["Work load Average/day "], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# no effect


# In[129]:


data.groupby(['Hit target']).count()


# In[131]:


data["Hit target"].value_counts()


# In[132]:


data.plot.scatter(x="Hit target", y="Absenteeism time in hours")


# In[133]:


plt.bar(x=data["Hit target"], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# equal propertion almost and nothing can be predicted here


# In[134]:


data.groupby(['Disciplinary failure']).count()


# In[137]:


data["Disciplinary failure"].value_counts()


# In[138]:


data.plot.scatter(x="Disciplinary failure", y="Absenteeism time in hours")


# In[139]:


plt.bar(x=data["Disciplinary failure"], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# ntohing here which means when they work all employess follow the rules, important to notice


# In[140]:


data.groupby(['Education']).count()


# In[141]:


data["Education"].value_counts()


# In[142]:


data.plot.scatter(x="Education", y="Absenteeism time in hours")


# In[143]:


plt.bar(x=data["Education"], height=data["Absenteeism time in hours"],width=0.8)


# In[144]:


# this clearly shows if you hire more employees who have studied till high school, u will get more out of it.( as per predicted it).
# but also then most emloyees are more primary, and doing their work so its data analysis could lead to more results, as less data available for 
# 3 otherr classes


# In[145]:


data.groupby(['Son']).count()


# In[146]:


data["Son"].value_counts()


# In[147]:


data.plot.scatter(x="Son", y="Absenteeism time in hours")


# In[148]:


plt.bar(x=data["Son"], height=data["Absenteeism time in hours"],width=0.8)


# In[149]:


# It can be said ppl with more childrn above 2 tend to be more hard working compared to less children which is understable( less data to analyze)
# Also not too concreate as many still worl( less absent ) but its just a comparision( which related to old age emplyees more efficent then mid thirties)


# In[150]:


data.groupby(['Social drinker']).count()


# In[151]:


data["Social drinker"].value_counts()


# In[152]:


data.plot.scatter(x="Social drinker", y="Absenteeism time in hours")


# In[153]:


plt.bar(x=data["Social drinker"], height=data["Absenteeism time in hours"],width=0.8)


# In[154]:


# nothing can be deciphered here


# In[155]:


data.groupby(['Social smoker']).count()


# In[156]:


data["Social smoker"].value_counts()


# In[157]:


data.plot.scatter(x="Social smoker", y="Absenteeism time in hours")


# In[158]:


plt.bar(x=data["Social smoker"], height=data["Absenteeism time in hours"],width=0.8)


# In[159]:


# This clearly suggest many ppl dont smoke, and even some do doesnt have adverse affect on their work as it can be seen here.


# In[160]:


data.groupby(['Pet']).count()


# In[161]:


data["Pet"].value_counts()


# In[162]:


data.plot.scatter(x="Pet", y="Absenteeism time in hours")


# In[163]:


plt.bar(x=data["Pet"], height=data["Absenteeism time in hours"],width=0.8)


# In[164]:


data.groupby(['Weight']).count()


# In[165]:


data["Weight"].value_counts()


# In[166]:


data.plot.scatter(x="Weight", y="Absenteeism time in hours")


# In[167]:


plt.bar(x=data["Weight"], height=data["Absenteeism time in hours"],width=0.8)


# In[ ]:


# Here we have a clear pattern unfit ppl both ( high and low wt ) above 89 kgs andbelow 68 kgswith couple of also fit( for some wts which are healthy)


# In[168]:


data.groupby(['Height']).count()


# In[169]:


data["Height"].value_counts()


# In[170]:


data.plot.scatter(x="Height", y="Absenteeism time in hours")


# In[171]:


plt.bar(x=data["Height"], height=data["Absenteeism time in hours"],width=0.8)


# In[173]:


# here it can be said ppl with high height tend to be more absent and irregular, ( ppl with 173 to 195 or below, no deata available )
# are pretty regular in their work


# In[174]:


data.groupby(['Body mass index']).count()


# In[175]:


data["Body mass index"].value_counts()


# In[176]:


data.plot.scatter(x="Body mass index", y="Absenteeism time in hours")


# In[177]:


plt.bar(x=data["Body mass index"], height=data["Absenteeism time in hours"],width=0.8)


# In[178]:


# nothing here ( even though here ppl with less weight and more height makes sense and less BMI ppl tend to be more absent)


# In[179]:


corrM = data.corr()


# In[180]:


corrM


# In[181]:


import seaborn as sns
sns.set(rc = {'figure.figsize':(15,8)})
ax = sns.heatmap(data.corr(), annot=True)


# In[182]:


pair = {}


# In[183]:


for i in range(0,20):
    columnName = data.columns[i]
    if i == 9:
        continue
    else:
        p = data['Absenteeism time in hours'].corr(data[columnName])
    if p>0:
        pair[columnName] = p
    print("Column Name:",columnName)
    print("Correlation:",p)


# In[184]:


pair


# In[185]:


# Certain statistical procedure to understand some features and get more analysis on them


# In[186]:


# TRAINING IN A MODEL TO UNDERSTAND THE FEATURES AND THEIR IMPORTANCE BETTER, INITIALLY USING ALL FEATURES, ,'Work load Average/day '


# In[187]:


X = data.drop(['Absenteeism time in hours','Work load Average/day '],axis=1)
Y = data['Absenteeism time in hours']


# In[188]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=42)


# In[189]:


from xgboost import XGBClassifier


# In[190]:


xgb_model = XGBClassifier(random_state = 0 ,use_label_encoder=True)
xgb_model.fit(X_train, Y_train)

print("Feature Importances : ", xgb_model.feature_importances_) 


# In[ ]:


# interms of feature importance - reason,transportation,smoker,drinker,son, service time


# In[201]:


data.columns


# In[191]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[192]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[193]:


lm.fit(X_train,Y_train)


# In[194]:


predictions = lm.predict(X_test)
from sklearn import metrics
y_pred = lm.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[195]:


# ML procedure


# In[ ]:





# In[ ]:




