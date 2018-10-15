"""this script cleans the data for the kaggle house pricing competition
"""
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import Imputer
import numpy as np

input_train_data = './train.csv'
input_test_data = './test.csv'

""" Cols with na values
MasVnrType
GarageQual
GarageFinish
Fence
Electrical
BsmtQual
MasVnrArea
BsmtCond
LotFrontage
BsmtExposure
MiscFeature
GarageYrBlt
PoolQC
BsmtFinType2
BsmtFinType1
GarageType
Alley
FireplaceQu
GarageCond
"""

""" Cols with string value types
Alley
BldgType
BsmtCond
BsmtExposure
BsmtFinType1
BsmtFinType2
BsmtQual
CentralAir
Condition1
Condition2
Electrical
ExterCond
ExterQual
Exterior1st
Exterior2nd
Fence
FireplaceQu
Foundation
Functional
GarageCond
GarageFinish
GarageQual
GarageType
Heating
HeatingQC
HouseStyle
KitchenQual
LandContour
LandSlope
LotConfig
LotShape
MSZoning
MasVnrType
MiscFeature
Neighborhood
PavedDrive
PoolQC
RoofMatl
RoofStyle
SaleCondition
SaleType
Street
Utilities
"""

def clean_up():
	# read in data and check input values
	df_train = pd.read_csv(input_train_data)
	df_test = pd.read_csv(input_test_data)
	train_objs_num = len(df_train)
	df = pd.concat(objs=[df_train, df_test], axis=0)
	cols = list(df.columns.values)
	print("the shape of the train input data is {}".format(df_train.shape))
	print("the shape of the test input data is {}".format(df_test.shape))
	print("The first 3 example lines of train data:")
	print(df_train[:3])

	# use one-hot encoding to encode string cols
	str_cols = [col for col in cols if is_string_dtype(df[col])]
	df = pd.get_dummies(df, columns=str_cols, dummy_na=True)
	print("After one-hot encoding, the first 3 example lines data :")
	print(df[:3])

	# using inputer to fill the na values with the mean of that col
	num_cols_with_na = df.columns[df.isna().any()].tolist()
	print("The following cols contains missing data: {}".format(num_cols_with_na))
	print("We are using a inputation strategy of filling by mean. We will fill the na value with the mean on that particular col")
	imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)
	imputed_df = pd.DataFrame(imputer.fit_transform(df))
	imputed_df.columns = df.columns
	imputed_df.index = df.index
	df = imputed_df

	# split back into train and test set
	df_train = df[:train_objs_num]
	df_test = df[train_objs_num:]
	print('After processing, we have train set size: {}'.format(df_train.shape))
	print('After processing, we have test set size: {}'.format(df_test.shape))
	print("The first 3 example lines of train data:")
	print(df_train[:3])

	# drop the Saleprice for the testing data, as it is what we are trying to predict
	df_test.drop(columns=['SalePrice'], inplace=True, axis=1)

	# save the data back into files
	df_train.to_csv(open('train_processed.csv', 'w'))
	df_test.to_csv(open('test_processed.csv', 'w'))

if __name__ == '__main__':
	clean_up()