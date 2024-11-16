# Preprocessing

Before training, we should analyse the data and do some preprocessing.

* Feature scale
* Remove and imputing missing values
* Convert categorical data into numeric
* Select relevant features 

## Feature scaling

Not all models require it, but it is recommended that all features are numeric and within the same range.

```
from sklearn.preprocessing import StandardScaler
 sc = StandardScaler
 sc.fit(X_train)
 X_train_std = sc.transform(X_train)
 X_test_std = sc.transform(X_test)
 ```

## Clean up data

By default we load data from a CSV file using pandas. It will be stored in a data frame.

```
import pandas as pd
df = pd.read_csv('data.csv')
df
```

### Drop data

Check how many empty values there are per column.

```
df.isnull().sum()
```

Remove rows with missing data

```
df.dropna()
```

Remove columns with missing data

```
df.dropna(axis=1)
```

Drop rows with a least a certain number of empty values

```
df.dropna(thresh=4)
```

### Imputing missing values

In this example, missing data will be populate with the average of values.

```
import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X_train)
X_train = imp_mean.transform(X_train)
X_test = imp_mean.transform(X_test)
```
