# In this project we will learn about what kinds of steps and challenges a data science go through for any problem in bbig comapnies.
# We will predict price of property in this project. We will build website using html , css ,java script which can predict home price for us 
# This project include concept of data science like Data cleaning, Feature engineering, One Hot Encoding, Outliner Detection, Dimensionality Reduction, GridSearchCV

import pandas as pd 
df = pd.read_csv("A:/machine learning/project 1/bengaluru_house_prices.csv")
print(df.head())

# ____________ 1) Data cleaning :  ____________ 

df = df.drop(['area_type','availability','society','balcony'],axis='columns')

print(df.isnull().sum())
df = df.dropna()
print(df.shape)


# ____________ 2) Feature engineering :  ____________ 

print(df['size'].unique())      # size column contain 1st word as number of bedroom. It represent one type of size of house differently like '2 BHK' & '2 Bedrooms'.

df['BHK'] = df['size'].apply(lambda x : int(x.split(' ')[0]))
df = df.drop('size',axis='columns')
print(df.head())

print(df['total_sqft'].unique())   # total_sqft column contain value in range and in other unit(rather than sqrt) also .
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   

df.total_sqft = df.total_sqft.apply(convert_sqft_to_num)
df = df[df.total_sqft.notnull()]

df['price_per_sqft'] = round(df['price']*100000 / df['total_sqft'],2)


# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations

print(len(df['location'].unique()))   # we have 1265 unique location

df.location = df.location.apply(lambda x: x.strip())

location_stats = df['location'].value_counts(ascending=False)

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount.
# This method is called Dimensionality Reduction .

df.location = df.location.apply(lambda x : 'other' if x in location_stats[location_stats<=10] else x)
print(len(df.location.unique()))


# ____________ 3) Outlier Removal Using Business Logic :  ____________ 

# Normally square ft per bedroom is 300. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier.
#  We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft 
print(df.shape)
df = df[~(df['total_sqft']/df.BHK < 300) ]
print(df.shape)

print(df.price_per_sqft.describe())

# Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices.
# mean means average price for price_per_sqrt and standard deviation means a quantity expressing by how much the members of a group differ from the mean value for the group.
# so value can be in the range of mean-std to mean+std. value beyond this range are outliers
# price differ for every location so we will find average price for every location and then will remove outlier from that.

import numpy as np

def remove_pps_outliers(df):
    df_out = pd.DataFrame()

    for key,subdf in df.groupby('location'):     # here key means location and subdf means dataframe which contain key location only 
        m = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduce = subdf[(subdf.price_per_sqft >= (m-std)) & (subdf.price_per_sqft <= (m+std))]
        df_out = pd.concat( [df_out, reduce],ignore_index=True)

    return df_out

df = remove_pps_outliers(df)

# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

import matplotlib.pyplot as plt 

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]

    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")

    plt.legend()
    plt.show()

plot_scatter_chart(df,"Rajaji Nagar")
plot_scatter_chart(df,"Hebbal")

# We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area).
def remove_bhk_outliers(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:                
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    
    return df.drop(exclude_indices,axis='index')

# Here , We have to add stats in the if condition because if for some location bhk starts from 3, then 2 bhk (bhk - 1) rows are not present in the data. so to ensure that the weather stats variable is present or not.

df = remove_bhk_outliers(df)
print(df.shape)

plot_scatter_chart(df,"Rajaji Nagar")
plot_scatter_chart(df,"Hebbal")

# if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error 
df = df[df.bath<df.BHK+1]

print(max(df.BHK))
print(max(df.bath))
