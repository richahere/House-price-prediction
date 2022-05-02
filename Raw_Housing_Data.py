import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing
Raw_Housing_Data=pd.read_csv("1. Regression - Module - (Housing Prices).csv")
# # print(Raw_Housing_Data['Sale Price'].head(10))
# # print(Raw_Housing_Data['Sale Price'].describe())
# # here we see that mean and median value (50%) is not same. So our data isnot normally distributed.

# plt.scatter(x=Raw_Housing_Data['ID'],y=Raw_Housing_Data['Sale Price'])
# plt.show()

# # to plot outlier using box whisker plot having null values it can't be done using matplotlib but we can use seaborn library
# # sns.boxplot(x=Raw_Housing_Data['Sale Price'])

# # in order to treat outlier we will remove lowest value of the data with either min value or lower limit and highest 
# # value with either upper limit or max value.
# iqr=Raw_Housing_Data['Sale Price'].quantile(.75)-Raw_Housing_Data['Sale Price'].quantile(.25)
# upper_limit=Raw_Housing_Data['Sale Price'].quantile(.75)+1.5*iqr
# lower_limit=Raw_Housing_Data['Sale Price'].quantile(.25)-1.5*iqr
# # We will calculate inter quartile range here to find lower limit as well as higher limit
# # lower limit=q1-1.5*iqr and higher_limit=q3+1.5*iqr
# # now we will treat target variable outlier here
# def treat_outlier(data):
#     if(data>upper_limit):
#         return upper_limit
#     if(data<lower_limit):
#         return lower_limit

# Raw_Housing_Data['Sale Price']=Raw_Housing_Data['Sale Price'].apply(treat_outlier)
# print(Raw_Housing_Data['Sale Price'].describe())
# # now plot scatter plot and you'll see difference in plots.
# plt.scatter(x=Raw_Housing_Data['ID'], y=Raw_Housing_Data['Sale Price'])
# plt.show()

# # now that we have treated outlier we will treat null values using deletion by use of dropna funcation. dropna(Inplace=True,axis=0,subset-['name])
# Raw_Housing_Data['Sale Price'].dropna(inplace=True,axis=0)
# print(Raw_Housing_Data.info())

# # Indepenent variable treatment
# # now that we have treated target variable we'll trat independent variable now. And we'll start using 
# # missing value treatment using imputation of numerical value and we'll replace missing value with mean or median
# # for imputation we use science kit library 
# Raw_Housing_Data.info()
# numerical_variable=['No of Bathrooms', 'Flat Area (in Sqft)','Lot Area (in Sqft)','No of Floors','Area of the House from Basement (in Sqft)',
#                     'Latitude', 'Longitude', 'Living Area after Renovation (in Sqft)']
# imputer=SimpleImputer(missing_values=np.nan, strategy='median')
# Raw_Housing_Data[numerical_variable]=imputer.fit_transform(Raw_Housing_Data[numerical_variable])
# column=Raw_Housing_Data['Zipcode'].values.reshape(-1,1)
# imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
# Raw_Housing_Data['Zipcode']=imputer.fit_transform(column)
# Raw_Housing_Data.info()

# # zipcode isnot contatining valuable information as it is in numerical form. So we will convert it into object data type.
# Raw_Housing_Data['Zipcode']=Raw_Housing_Data['Zipcode'].astype(object)
# Raw_Housing_Data.info()

#   No of times visited is in form of object so we will convert it into numerial form
mapping={'None':0,
        'Once':1,
        'Twice':2,
        'Thrice':3,
        'Four':4
}
Raw_Housing_Data['No of Times Visited']=Raw_Housing_Data['No of Times Visited'].map(mapping)
print(Raw_Housing_Data['No of Times Visited'].unique())
# since year of renovation isnot giving usefull information here. So we'll derive usefull info using data transformation
# we will derive total time since renovation using renovated year and date house was sold
Raw_Housing_Data['ever_renovated']=np.where(Raw_Housing_Data['Renovated Year']==0,'No','Yes')
Raw_Housing_Data['purchase_year']=pd.DatetimeIndex(Raw_Housing_Data['Date House was Sold']).year
Raw_Housing_Data['renovated_year']=pd.DatetimeIndex(Raw_Housing_Data['Renovated Year']).year
Raw_Housing_Data['year_since_reonovation']=np.where(Raw_Housing_Data['ever_renovated']=='Yes',
                                        abs(Raw_Housing_Data['purchase_year']-Raw_Housing_Data['renovated_year']),0)
print(Raw_Housing_Data.head(20))

# now since we have derived year since renovation using date of purchase and year of renovavtion we'll drop these variable
# -------renovated year, date of purchase, purchase year
Raw_Housing_Data.drop(columns=['Date House was Sold','Renovated Year','renovated_year','purchase_year'],inplace=True)
print(Raw_Housing_Data.head())

# ID column isnot adding importance in our dataset so we will drop it
Raw_Housing_Data.drop(columns=['ID'],inplace=True)
print(Raw_Housing_Data.info())
# now we will work on condition of the house
# in order to find the frequency of th unique values in dataset we use funcation value_counts()
print(Raw_Housing_Data['Condition of the House'].value_counts())
# now we will see if there is any relation among independent ie conditon of the house and target variable
# we will use bar graph for that
# Raw_Housing_Data.groupby('Condition of the House')['Sale Price'].mean().sort_values().plot(kind='bar')
# Raw_Housing_Data.groupby('Waterfront View')['Sale Price'].mean().sort_values().plot(kind='bar')
# Raw_Housing_Data.groupby('Zipcode')['Sale Price'].mean().sort_values().plot(kind="bar")
# plt.show()

# now we will study about relationship among categorical and numerical variable using anova
# Two hypothesis--> NULL(There is no diff in mean)-->Alternate(There is diff in atleast one)
# two values F and P values F should be larger and p<0.05 for rejecting null hypothesis
# to implement anova use module statsmodels
# in order to implement this we have to do simple name change in this 
Raw_Housing_Data=Raw_Housing_Data.rename(columns={'Sale Price':'sale_price'})
Raw_Housing_Data=Raw_Housing_Data.rename(columns={'Condition of the House':'condition_of_the_house'})
Raw_Housing_Data=Raw_Housing_Data.rename(columns={'Waterfront View':'waterfront_view'})
# now we will find anova
mod=ols('sale_price ~ condition_of_the_house', data=Raw_Housing_Data).fit()
# now create anova table
anova_table=sm.stats.anova_lm(mod,type=2)
print(anova_table)
mod=ols('sale_price~waterfront_view',data=Raw_Housing_Data).fit()
anova_table=sm.stats.anova_lm(mod,type=2)
print(anova_table)
mod=ols('sale_price~ever_renovated',data=Raw_Housing_Data).fit()
anova_table=sm.stats.anova_lm(mod,type=2)
print(anova_table)
# above we got anova

# in regression model categorical data is not accepted so we will convert categorical data into numerical data
# we can do this by using pd.get_dummies() funcation. This whole process is called dummy variable
# if a categorical variable has n unoque values then total dummy variable willbe (n-1)
Raw_Housing_Data=pd.get_dummies(Raw_Housing_Data,columns=['condition_of_the_house','waterfront_view','ever_renovated'],drop_first=True)
print(Raw_Housing_Data.head(10))

# sometimes if the number of unique variable in any categorical variable is larger then we need to convert into bins because
# total dummy variable will be larger in this case
# to create bins we will use pd.cut funcation
Zip_table=Raw_Housing_Data.groupby('Zipcode').agg({'sale_price':'mean'}).sort_values('sale_price',ascending=True)
print(Zip_table.head())
# now create bin
Zip_table['Zipcode_Group']=pd.cut(Zip_table['sale_price'],bins=10,labels=['Zipcode_Group_0',
                                                                        'Zipcode_Group_1',
                                                                        'Zipcode_Group_2',
                                                                        'Zipcode_Group_3',
                                                                        'Zipcode_Group_4',
                                                                        'Zipcode_Group_5',
                                                                        'Zipcode_Group_6',
                                                                        'Zipcode_Group_7',
                                                                        'Zipcode_Group_8',
                                                                        'Zipcode_Group_9'],include_lowest=True)

Zip_table.drop(columns='sale_price',inplace=True)
Raw_Housing_Data=pd.merge(Raw_Housing_Data,Zip_table,left_on="Zipcode",how='left',right_index=True)
Raw_Housing_Data=Raw_Housing_Data.drop(columns='Zipcode')
Raw_Housing_Data=pd.get_dummies(Raw_Housing_Data,columns=['Zipcode_Group'],drop_first=True)
print(Raw_Housing_Data.head())
# now we will seperate dependent and independent variable Y=dependent(sale_price),X=independent variable(remainining variable)
# for this we will use iloc funcation iloc[:,0]
Y=Raw_Housing_Data.iloc[:,0]
X=Raw_Housing_Data.iloc[:,1:31]
print(X.head(5),Y.head(5))
# now we will split our data into train and test data
# train data is used by machine to learn. Machine is built on our trained data. In this output is known to us.
# test data is used to test our machine 
# to divide data into train and test data we use sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.3)

#  now we will scale our data using  from sklearn import preprocessing
scale=preprocessing.StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.fit_transform(X_train)

# our first model to predict sale price of house
# we will use mean sale price in this.
Raw_Housing_Data['mean_sales']=Raw_Housing_Data['sale_price'].mean()
print(Raw_Housing_Data['mean_sales'].head())
# but this sale price prediction is not 100% correct
# we can show this by plotting scatter plot
k=range(0,len(Raw_Housing_Data['sale_price']))
plt.figure(dpi=100)
plt.scatter(k,Raw_Housing_Data['sale_price'].sort_values(),color="red",label="Actual Sale Price")
plt.scatter(k,Raw_Housing_Data['mean_sales'].sort_values(),color="green",label="Mean Sale Price")
plt.xlabel("Fitted Points")
plt.ylabel("Sale Price")
plt.title("Overall Price")
plt.show()
# from here we can say that sale price and mean sale isnot same