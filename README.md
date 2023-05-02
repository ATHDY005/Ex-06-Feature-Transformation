# Ex-06-Feature-Transformation
## AIM

To read the given data and perform Feature Transformation process and save the data to a file.

## ALGORITHM

# STEP 1

Read the given Data

# STEP 2

Clean the Data Set using Data Cleaning Process

# STEP 3

Apply Feature Transformation techniques to all the feature of the data set

# STEP 4

Save the data to the file

# PROGRAM 
import pandas as pd
df=pd.read_csv('/content/Data_to_Transform.csv')
df.head()
df.isnull().sum()
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
sm.qqplot(df['Highly Positive Skew'], fit=True,line='45')
plt.show()
sm.qqplot(df['Highly Negative Skew'], fit=True,line='45')
plt.show()
sm.qqplot(df['Moderate Positive Skew'], fit=True,line='45')
plt.show()
sm.qqplot(df['Moderate Negative Skew'], fit=True,line='45')
plt.show()
df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'], fit=True,line='45') 
plt.show()
df['Highly Positive Skew']=1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'], fit=True,line='45')
plt.show()
df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew']) 
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer("yeo-johnson")
df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']])) 
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal")
df['Moderate Negative Skew']-pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'], fit=True,line='45')
plt.show()

# OUTPUT 

![image](https://user-images.githubusercontent.com/84709944/235635678-059ceb9c-6880-4941-91f8-8a8828d9055c.png)
![image](https://user-images.githubusercontent.com/84709944/235635716-b70fa840-314c-4db5-868c-f5ef10dd5e2c.png)
![image](https://user-images.githubusercontent.com/84709944/235636554-1c3c127f-80d9-44c0-9556-7d39114ba181.png)
![image](https://user-images.githubusercontent.com/84709944/235636630-d5b930f1-2a9b-421f-9fbc-058c79acbaca.png)
![image](https://user-images.githubusercontent.com/84709944/235636676-a1d87f0b-d556-4bac-9f3a-580f1ac982cf.png)
![image](https://user-images.githubusercontent.com/84709944/235636725-ecd3e53e-710b-4b2c-bf99-0b8f522b200a.png)
![image](https://user-images.githubusercontent.com/84709944/235636784-46cfbfe6-d4b9-4200-ae1c-8a8cfb56d633.png)
![image](https://user-images.githubusercontent.com/84709944/235636842-de010748-4847-4c32-8bf1-68a71ed2f847.png)
![image](https://user-images.githubusercontent.com/84709944/235636905-c1724530-c23c-44b9-a81a-e64adc8fdba3.png)
![image](https://user-images.githubusercontent.com/84709944/235636952-563dbbaf-6c39-4c02-a450-4789e8675ded.png)
![image](https://user-images.githubusercontent.com/84709944/235637008-f939370b-a9ff-4f87-8d15-cadd9697e2f5.png)

# RESULT

Hence the data was read was performed the Feature Transformation process and saved the data to a file.
