import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('house-prices-hw/train_hw.csv')
test_data = pd.read_csv('house-prices-hw/test_hw.csv')
test_ids = test_data['Id']
train_data = train_data.drop(columns=['Id'])
test_data = test_data.drop(columns=['Id'])

def remove_high_nan_cols(df, numeric_cols, categorical_cols, numeric_threshold=0.70, categorical_threshold=0.80):
    nan_percent = df.isnull().mean()
    cols_to_drop = []
    for col in df.columns:
        if col in numeric_cols and nan_percent[col] >= numeric_threshold:
            cols_to_drop.append(col)
        elif col in categorical_cols and nan_percent[col] >= categorical_threshold:
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop)
    return df, [col for col in numeric_cols if col not in cols_to_drop], [col for col in categorical_cols if col not in cols_to_drop]

def create_new_features(df):
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['HasPool'] = df['PoolArea'] > 0
    df['OverallQualitySF'] = df['OverallQual'] * df['TotalSF']
    df['GarageScore'] = df['GarageCars'] * df['GarageArea']
    df['HasFireplace'] = df['Fireplaces'] > 0
    df['HasGarage'] = df['GarageArea'] > 0
    df['OverallQual_HouseAge'] = df['OverallQual'] * df['HouseAge']
    df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['3SsnPorch']
    df['TotalBsmtFin'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df['YearSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['LivingAreaRatio'] = df['GrLivArea'] / (df['LotArea'] + 1e-6)
    df['TotalLot'] = df['LotFrontage'] * df['LotArea']
    return df

# Определение числовых и категориальных колонок
numeric_cols = train_data.select_dtypes(exclude=['object']).columns.drop('SalePrice')
categorical_cols = train_data.select_dtypes(include=['object']).columns

train_data = create_new_features(train_data)
test_data = create_new_features(test_data)

train_data, numeric_cols, categorical_cols = remove_high_nan_cols(train_data, numeric_cols, categorical_cols)
test_data = test_data[[col for col in train_data.columns if col != 'SalePrice']]

# Определение числовых и категориальных колонок
numeric_cols = train_data.select_dtypes(exclude=['object']).columns.drop('SalePrice')
categorical_cols = train_data.select_dtypes(include=['object']).columns

def impute_missing_values(df, numeric_cols, categorical_cols):
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = df[categorical_cols].fillna('None')
    categorical_imputer = SimpleImputer(strategy='most_frequent', fill_value='None')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    return df

train_data = impute_missing_values(train_data, numeric_cols, categorical_cols)
test_data = impute_missing_values(test_data, numeric_cols, categorical_cols)

y = np.log1p(train_data['SalePrice'])
X = train_data.drop(columns=['SalePrice'])

# Удаление выбросов по log(SalePrice) с IQR 1.5
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (y >= lower_bound) & (y <= upper_bound)
X = X[mask]
y = y[mask]

# One-Hot Encoding для категориальных признаков
X_encoded = pd.get_dummies(X, drop_first=True)
test_encoded = pd.get_dummies(test_data, drop_first=True)

# Выравнивание колонок
X_encoded, test_encoded = X_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# Сохранение индексов ключевых числовых колонок для полиномизации
key_numeric_cols = ['OverallQual', 'TotalSF', 'GrLivArea', 'GarageArea', 'TotalBath']
numeric_indices = [X_encoded.columns.get_loc(col) for col in key_numeric_cols if col in X_encoded.columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
test_scaled = scaler.transform(test_encoded)

# Добавление полиномиальных признаков для ключевых числовых колонок
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled[:, numeric_indices])
test_poly = poly.transform(test_scaled[:, numeric_indices])
X_scaled = np.hstack([X_scaled, X_poly])
test_scaled = np.hstack([test_scaled, test_poly])

# Обучение Lasso с кросс-валидацией
lasso = LassoCV(alphas=np.logspace(-4, 0, 100), cv=5, max_iter=10000)
lasso.fit(X_scaled, y)
lasso_scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
lasso_rmse = np.sqrt(-lasso_scores)
print(f"Средний RMSE на кросс-валидации (Lasso): {lasso_rmse.mean():.4f}")

y_pred_log = lasso.predict(test_scaled)
y_pred = np.expm1(y_pred_log)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred})
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно сохранен!")