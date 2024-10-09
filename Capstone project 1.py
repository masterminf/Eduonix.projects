#!/usr/bin/env python
# coding: utf-8

# In[84]:


#Importing all neccessary library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


# In[85]:


#importing data file
df = pd.read_csv("roadAccStats13-16(DF).csv",encoding='latin-1')
df1 = pd.read_csv("Details_of_road_accident_deaths_by_situation_state_2014(DF1).csv",encoding="latin-1")
df2 = pd.read_csv("Persons_killed_due_to_Non-use_of_Safety_Device_2016(DF2).csv",encoding="latin-1")
df3 = pd.read_csv("Copy of accidents03-16(DF3).csv",encoding="latin-1")
df4 = pd.read_csv("laneAccidents(DF4).csv",encoding="latin-1")
df5 = pd.read_csv("reasonOfAccident(DF5).csv",encoding="latin-1")
df6 = pd.read_csv("typeOfVehicle(DF6).csv",encoding="latin-1")
df7 = pd.read_csv("timeOfOccurence(DF7).csv",encoding="latin-1")


# In[86]:


df.head()


# In[87]:


df.isnull().sum()


# In[88]:


df['State/UT-Wise Total Number of Road Accidents during - 2013']= df['State/UT-Wise Total Number of Road Accidents during - 2013'].fillna(df['State/UT-Wise Total Number of Road Accidents during - 2013'].mean())
df['Total Number of Accidents Per Lakh Population - 2013']= df['Total Number of Accidents Per Lakh Population - 2013'].fillna(df['Total Number of Accidents Per Lakh Population - 2013'].mean()) 
df['Total Number of Accidents Per Lakh Population - 2014']= df['Total Number of Accidents Per Lakh Population - 2014'].fillna(df['Total Number of Accidents Per Lakh Population - 2014'].mean()) 
df['Total Number of Accidents Per Lakh Population - 2015']= df['Total Number of Accidents Per Lakh Population - 2015'].fillna(df['Total Number of Accidents Per Lakh Population - 2015'].mean()) 
df['Total Number of Accidents Per Lakh Population - 2016']= df['Total Number of Accidents Per Lakh Population - 2016'].fillna(df['Total Number of Accidents Per Lakh Population - 2016'].mean()) 
df['Total Number of Road Accidents per 10,000 Vehicles - 2013']= df['Total Number of Road Accidents per 10,000 Vehicles - 2013'].fillna(df['Total Number of Road Accidents per 10,000 Vehicles - 2013'].mean())
df['Total Number of Road Accidents per 10,000 Km of Roads - 2013']= df['Total Number of Road Accidents per 10,000 Km of Roads - 2013'].fillna(df['Total Number of Road Accidents per 10,000 Km of Roads - 2013'].mean()) 


# In[89]:


df.isnull().sum()


# In[90]:


df.dtypes


# In[91]:


# Que 1 -The percentage of road accidents during all the years.

total_no_of_accidents = df[['State/UT-Wise Total Number of Road Accidents during - 2013',
                           'State/UT-Wise Total Number of Road Accidents during - 2014',
                          'State/UT-Wise Total Number of Road Accidents during - 2015',
                          'State/UT-Wise Total Number of Road Accidents during - 2016']].sum()
years = ['2013','2014','2015','2016']
#plotting in graphical reprentation
plt.figure(figsize=(10,5))
plt.plot(years, total_no_of_accidents, linestyle='--',color='r')
plt.xlabel('year') 
plt.ylabel('Total number of accidents')
plt.title('Total number of accidents from 2013-2016') 
plt.grid(True)
plt.show



# In[92]:


# Que 2 - Mean Accidents per 1L population for each year.

mean_for_2013 =df['Total Number of Accidents Per Lakh Population - 2013'].mean()
mean_for_2014 =df['Total Number of Accidents Per Lakh Population - 2014'].mean()
mean_for_2015 =df['Total Number of Accidents Per Lakh Population - 2015'].mean()
mean_for_2016 =df['Total Number of Accidents Per Lakh Population - 2016'].mean()


# In[93]:


print(mean_for_2013)
print(mean_for_2014)
print(mean_for_2015)
print(mean_for_2016)


# In[99]:


print(df.columns)


# In[129]:


# Que 3 -. The highest number of accident states and least number of accident states.
#first we have to group data by states and number of accidents
accident_summary = df.groupby('States/UTs')[['State/UT-Wise Total Number of Road Accidents during - 2013',
                     'State/UT-Wise Total Number of Road Accidents during - 2014',
                     'State/UT-Wise Total Number of Road Accidents during - 2015',
                     'State/UT-Wise Total Number of Road Accidents during - 2016']].sum()


# In[130]:


accident_summary


# In[ ]:


# from the above we can say that highest no of accidents are in Maharashtra and lowest are in Daman and Div


# In[102]:


# Que 4 - Offenders and victims who died according to gender as well the as the total deaths

df1.head()


# In[108]:





# In[139]:


# grouping deaths by gender for victims
victims_deaths_by_gender = df1.groupby('Year')[['Victims Died_Male','Victims Died_Transgender','Victims Died_Female']].sum()


# In[140]:


victims_deaths_by_gender


# In[141]:


## grouping deaths by gender for offenders
offenders_deaths_by_gender = df1.groupby('Year')[['Offenders (Driver/Pedestrian) Died_Male','Offenders (Driver/Pedestrian) Died_Female','Offenders (Driver/Pedestrian) Died_Transgender']].sum()


# In[142]:


offenders_deaths_by_gender


# In[144]:


# total offenders and victims
total_offenders = df1['Offenders (Driver/Pedestrian) Died_Total'].sum()
total_victims = df1['Victims Died_Total'].sum()
print(total_offenders)
print(total_victims)


# In[145]:


# Que 5- Percentage of Deaths occurring due to non-wearing of helmets between male and fenale
df2.head()


# In[146]:


# calculating total male and total male deaths
male_deaths = df2['Non-wearing of Helmet - Male'].sum()
female_deaths = df2['Non-wearing of Helmet - Female'].sum()
total_deaths = df2['Non-wearing of Helmet - Total'].sum()

print(male_deaths)
print(female_deaths)
print(total_deaths)


# In[147]:


# now calculate percentage
percentage_male =(male_deaths/total_deaths)*100
percentage_female =(female_deaths/total_deaths)*100

print(percentage_male)
print(percentage_female)


# In[148]:


# que 6 - The number of accidents happening per state from the year 2003 to 2016.
df3.head()


# In[151]:


df3['Total_accidents']=df3[['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']].sum()


# In[153]:


number_of_accidents_per_state = df3.groupby('ï»¿States/Uts')['Total_accidents'].sum()


# In[154]:


print(number_of_accidents_per_state)


# In[155]:


# Que 7 - Number of ACCIDENTS for 1,2,3,4 LANE per 1L population of resp. state.
df4.head()


# In[163]:


# Number of ACCIDENTS for 1,2,3,4 LANE per 1L population of resp. state.

Number_of_ACCIDENTS = df4.groupby('State/UT')[['Single Lane - Accident - 2014 per 1L people','Two Lanes - Accident - 2014 per 1L people','4 Lanes with Median - Accident - 2014 per 1L people']].sum()


# In[164]:


print(Number_of_ACCIDENTS)


# In[165]:


# Que 8-  Number of people INJURED for 1,2,3,4 type of lane per 1L population of resp. State.
Number_of_people_INJURED = df4.groupby('State/UT')[['Single Lane - Injured - 2014 per 1L people','3 Lanes or more w.o Median - Injured - 2014 per 1L people','4 Lanes with Median - Injured - 2014 per 1L people']].sum()


# In[166]:


print(Number_of_people_INJURED)


# In[170]:


# Que 9 - Number of people KILLED for 1,2,3,4 LANES per 1L population of resp. States.
Number_of_people_KILLED = df4.groupby('State/UT')[['Single Lane - Killed - 2014 per 1L people','3 Lanes or more w.o Median - Killed - 2014 per 1L people','4 Lanes with Median - Killed - 2014 per 1L people']].sum()


# In[171]:


print(Number_of_people_KILLED)


# In[172]:


#Que 10 - Number of Accidents, people KILLED, INJURED on SINGLE LANE per 1L population.

single_lane_data = df4.groupby('State/UT')[['Single Lane - Accident - 2014 per 1L people','Single Lane - Killed - 2014 per 1L people','Single Lane - Injured - 2014 per 1L people']].sum()


# In[173]:


print(single_lane_data)


# In[178]:


# Que 11 - Number of accidents, people INJURED, KILLED on DOUBLE LANE per 1L population.
double_lane_data = df4.groupby('State/UT')[['Two Lanes - Accident - 2014 per 1L people']].sum()


# In[179]:


print(double_lane_data)


# In[180]:


# Que 12 - Number of accidents, people INJURED, KILLED on THREE LANE per 1L population.

lane_3_data = df4.groupby('State/UT')[['3 Lanes or more w.o Median - Killed - 2014 per 1L people','3 Lanes or more w.o Median - Injured - 2014 per 1L people']].sum()


# In[181]:


print(lane_3_data)


# In[182]:


# Que 13 -Number of accidents, people INJURED, KILLED on FOUR LANE per 1L population.

lane_4_data = df4.groupby('State/UT')[['4 Lanes with Median - Accident - 2014 per 1L people','4 Lanes with Median - Killed - 2014 per 1L people','4 Lanes with Median - Injured - 2014 per 1L people']].sum()


# In[183]:


print(lane_4_data)


# 

# In[ ]:





# In[184]:


# Que 14 - Total Number of INJURED, KILLED, ROAD ACCIDENTS irrespective of lanes per 1L population of resp. State.
df4.head()


# In[191]:


# Total Number of INJURED, KILLED, ROAD ACCIDENTS irrespective of lanes per 1L population of resp. State. 

print(df4[['State/UT','Single Lane - Injured - 2014 per 1L people','3 Lanes or more w.o Median - Injured - 2014 per 1L people','4 Lanes with Median - Injured - 2014 per 1L people']])


# In[189]:





# In[192]:


# Que 15 - Number of people KILLED for each different REASON per 1L population of that state.
df5.head()


# In[194]:


# number of people  killed
people_killed = df5.groupby('States/UTs')[['Fault of Driver-Number of Persons-Killed - 2014 per 1L people','Other causes/causes not known-Number of Persons-Killed - 2014 per 1L people','Falling of boulders-Number of Persons-Killed - 2014 per 1L people']].sum()


# In[195]:


print(people_killed)


# In[196]:


#Que 16 -number of people INJURED for each reason per 1L people of that state.
people_injured = df5.groupby('States/UTs')[['Fault of Driver-Number of Persons-Injured - 2014 per 1L people','Falling of boulders-Number of Persons-Injured - 2014 per 1L people']].sum()


# In[197]:


print(people_injured)


# In[198]:


#Que 17 - Number of ACCIDENTS for each reason per 1L people of that state
number_of_accident = df5.groupby('States/UTs')[['Fault of Driver-Total No. of Road Accidents - 2014 per 1L people','Fault of Driver of other vehicles-Total No. of Road Accidents - 2014 per 1L people','Other causes/causes not known-Total No. of Road Accidents - 2014 per 1L people']].sum()


# In[199]:


print(number_of_accident)


# In[202]:


#Que no 18 -Total number of ROAD ACCIDENTS, INJURIES, DEATHS due to FAULT OF THE DRIVER per 1L population of that state.
fault_of_driver = df5.groupby('States/UTs')[['Fault of Driver-Total No. of Road Accidents - 2014 per 1L people','Fault of Driver-Number of Persons-Killed - 2014 per 1L people','Fault of Driver-Number of Persons-Injured - 2014 per 1L people']].sum()


# In[203]:


print(fault_of_driver)


# In[204]:


# Que 19 - Total number of ROAD ACCIDENTS, INJURIES, DEATHS due to the FAULT OF DRIVER'S FROM OTHER VEHICLES per 1L people of that state.
FAULT_OF_DRIVER_FROM_OTHER_VEHICLES = df5.groupby('States/UTs')[['Fault of Driver of other vehicles-Total No. of Road Accidents - 2014 per 1L people']].sum()


# In[205]:


print(FAULT_OF_DRIVER_FROM_OTHER_VEHICLES)


# In[268]:


# Que 20 - Total number of ROAD ACCIDENTS, INJURIES, DEATHS due to the FAULT OF PEDESTRIANS per 1L people of that state
df8 = pd.read_csv('location.csv',encoding='latin-1')
df8.columns


# In[269]:


df8.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[208]:


df7.head()


# In[ ]:





# In[ ]:





# In[229]:


# Que 27 - Number of Accidents happening in DAY and NIGHT TIME for 2014 and 2016.
accidents_in_day_2014 = (df7[('06-900hrs - Day - 2014')]+df7[('09-1200hrs - Day - 2014')]+df7[('12-1500hrs - Day - 2014')]+df7[('15-1800hrs - Day - 2014')]).sum()


# In[227]:


accidents_in_day_2014


# In[231]:


accidents_in_day_2016 =(df7[('06-900hrs - (Day) - 2016')]+df7[('09-1200hrs - (Day) - 2016')]+df7[('12-1500hrs - (Day) - 2016')]+df7[('15-1800hrs - (Day) - 2016')]).sum()


# In[232]:


accidents_in_day_2016


# In[234]:


Number_of_Accidents_happening_in_DAY = accidents_in_day_2014 + accidents_in_day_2016
print(Number_of_Accidents_happening_in_DAY)


# In[236]:


accidents_in_night_2014 = (df7['18-2100hrs - Night - 2014']+df7['21-2400hrs - Night - 2014']+df7['00-300hrs - Night - 2014']+df7['03-600hrs - Night - 2014']).sum()
print(accidents_in_night_2014)


# In[238]:


accidents_in_night_2016 = (df7['18-2100hrs - (Night) - 2016']+df7['21-2400hrs - (Night) - 2016']+df7['00-300hrs - (Night) - 2016']+df7['03-600hrs - (Night) - 2016']).sum()
print(accidents_in_night_2016)


# In[240]:


Number_of_Accidents_happening_in_NIGHT = accidents_in_night_2014+ accidents_in_night_2016
print(Number_of_Accidents_happening_in_NIGHT)


# In[241]:


#Que 26 - Total accidents, fatal accidents, killed and injured for each state per 1L people of that state.
df6.head()


# In[258]:


Total_accidents = df6.groupby('States/UTs')[['Two-Wheelers - Number of Road Accidents - Total - 2014 per 1L people','Other Vehicles/Objects - Number of Road Accidents - Total - 2014 per 1L people']].sum()
print(Total_accidents)


# In[261]:


Fatal_accidents = df6.groupby('States/UTs')[['Two-Wheelers - Number of Road Accidents - Fatal - 2014 per 1L people','Other Vehicles/Objects - Number of Road Accidents - Fatal - 2014 per 1L people']].sum()
print(Fatal_accidents)


# In[262]:


total_killed = df6.groupby('States/UTs')[['Two-Wheelers - Number of Persons - Killed - 2014 per 1L people','Other Vehicles/Objects - Number of Persons - Killed - 2014 per 1L people']].sum()
print(total_killed )


# In[264]:


total_injured = df6.groupby('States/UTs')[['Other Vehicles/Objects - Number of Persons - Injured - 2014 per 1L people','Two-Wheelers - Number of Persons - Injured - 2014 per 1L people']].sum()
print(total_injured)


# In[248]:


#Que 25 - . Number of Persons Killed for each vehicle type per 1L people of that state.

Number_of_Persons_Killed = (df6['Two-Wheelers - Number of Persons - Killed - 2014 per 1L people']+df6['Other Vehicles/Objects - Number of Persons - Killed - 2014 per 1L people']).sum()
print(Number_of_Persons_Killed)


# In[249]:


# Que 24 - Number of Total Accidents for each vehicle type per 1L people of that state

Total_accidents = (df6['Two-Wheelers - Number of Road Accidents - Total - 2014 per 1L people']+df6['Other Vehicles/Objects - Number of Road Accidents - Total - 2014 per 1L people']).sum()
print(Total_accidents)


# In[ ]:




