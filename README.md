# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("C:/Users/acer/Downloads/bmi.csv")
df

```

<img width="233" height="299" alt="image" src="https://github.com/user-attachments/assets/7b963d2d-425d-4108-a738-846a087a1b0d" />

```

df.head()

```

<img width="264" height="331" alt="image" src="https://github.com/user-attachments/assets/2a4a6c91-d953-42a6-8178-9459c063249c" />


```

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Hs','Ws']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)


```

<img width="392" height="316" alt="image" src="https://github.com/user-attachments/assets/2b132e35-6be0-4b55-afde-05ba4d3c8af8" />


```

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df[['Hm','Wm']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)

```

<img width="426" height="249" alt="image" src="https://github.com/user-attachments/assets/15be0965-488b-456a-9290-513a93b07bda" />

```

import pandas as pd
df = pd.read_csv("C:/Users/acer/Downloads/titanic_dataset.csv")
df.columns

```

<img width="502" height="68" alt="image" src="https://github.com/user-attachments/assets/b3e7e397-7b06-48a7-a32a-ab16f6d8c7ba" />

```

df

```

<img width="786" height="344" alt="image" src="https://github.com/user-attachments/assets/90398326-f6c4-471f-be32-684909a95811" />

```


df.isnull().sum()

```

<img width="181" height="196" alt="image" src="https://github.com/user-attachments/assets/e23fc758-5491-4fbd-8514-6f28892f2817" />

```

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
# Input and output
X = df[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = df['Survived']
# Feature Selection
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

<img width="475" height="55" alt="image" src="https://github.com/user-attachments/assets/a756f972-a01e-4d3e-b4b7-70ee9031bdc8" />

```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```

<img width="323" height="125" alt="image" src="https://github.com/user-attachments/assets/b6547093-0629-4115-b36f-2ac396b4714f" />


```

tips.time.unique()

```

<img width="436" height="48" alt="image" src="https://github.com/user-attachments/assets/4b80f4f2-8450-4c36-a426-4047ce93c7f5" />

```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```

<img width="165" height="77" alt="image" src="https://github.com/user-attachments/assets/c30a41c0-4ebe-4e31-a482-c89215fb1cf4" />


```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```

<img width="277" height="51" alt="image" src="https://github.com/user-attachments/assets/e0f11767-c9f9-43c1-8736-7e1068429b3b" />


# RESULT:
 Thus the program is executed successfully.
