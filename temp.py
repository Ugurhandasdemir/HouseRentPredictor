import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor
from math import sqrt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt

data = pd.read_csv('/home/ugo/Documents/Python/rent_log/rent_dataset/House_Rent_Dataset.csv')


data = data.rename(columns={
    'Posted On': 'date',
    'BHK': 'bhk',
    'Rent': 'rent',
    'Size': 'size',
    'Floor': 'floor',
    'Area Type': 'area_type',
    'Area Locality': 'locality',
    'City': 'city',
    'Furnishing Status': 'furnish',
    'Tenant Preferred': 'tenant',
    'Bathroom': 'bath',
    'Point of Contact': 'contact',
})

#gereksiz verileri çıkar
data = data[(data['area_type'] != 'Built Area') & (data['contact'] != 'Contact Builder')] 





print(data.sort_values(by='rent', ascending=False).head()) #1837 index aykırı

print(data.bath.sort_values().tail(10)) #4185 index aykırı



warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(10, 5))
sns.boxplot(x='city', y='rent', data=data, showfliers=False, palette='Set2', linewidth=1.2)
sns.stripplot(x='city', y='rent', data=data, hue='city', dodge=False, palette='Set2', jitter=0.25, size=8, alpha=0.9)
plt.legend().remove()
plt.xticks(rotation=45, fontsize=10)
plt.title('Şehire Göre Kira', fontsize=14, weight='bold')
plt.ylabel('Kira', fontsize=12)
plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.show()



# aykırı değerleri sil
data = data.drop(index=[1837], errors='ignore')
upper_limit = data['rent'].quantile(0.99) #en büyük %1 bul
data = data[data['rent'] <= upper_limit]
data = data.drop(index=[4185], errors='ignore') 

# gereksiz sutunları çıkar
data = data.drop(columns=['date', 'locality', 'tenant'])

# int değere çevir
data['furnish'] = data['furnish'].replace({'Unfurnished': 0, 'Semi-Furnished': 0.5, 'Furnished': 1})

for old, new in {'Ground': '0', 'Upper': '-1', 'Lower': '-2'}.items():
    data['floor'] = data['floor'].str.replace(old, new)
data['max_floor'] = data['floor'].apply(lambda x: int(x.split(' ')[-1]))
data['floor'] = data['floor'].apply(lambda x: int(x.split(' ')[0]))

# one-hot encoding 
data = pd.get_dummies(data, columns=['city', 'contact', 'area_type'], drop_first=True)

fig, ax = plt.subplots(figsize=(12,  6))
sns.heatmap(data.corr().round(2), annot=True, ax=ax)
fig.tight_layout()


# XGBoost Gridataearch ile en iyi parametreyi seç
features = [
    'size', 'area_type_Super Area', 'bath', 'furnish', 'max_floor', 'floor',
    'contact_Contact Owner', 'bhk', 'city_Chennai', 'city_Delhi',
    'city_Hyderabad', 'city_Kolkata', 'city_Mumbai'
]

x = data[features]
y = np.log1p(data['rent'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
}


model1 =  XGBRegressor(random_state=42)

grid_search = GridSearchCV(model1, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

grid_search.fit(x_train, y_train)  

best_param = grid_search.best_params_
print("Best Params:",best_param)

model2 = XGBRegressor(**best_param, random_state=42)
model2.fit(x_train, y_train)

pred_train = np.expm1(model2.predict(x_train))
pred_test = np.expm1(model2.predict(x_test))
y_train_orig = np.expm1(y_train)
y_test_orig = np.expm1(y_test)

print('Train')
print(f'MSE: {mean_squared_error(y_train_orig, pred_train)}')
print(f'MAPE: {mean_absolute_percentage_error(y_train_orig, pred_train)}')
print(f'RSQ: {r2_score(y_train_orig, pred_train)}')
print(f'RMSE: {sqrt(mean_squared_error(y_train_orig, pred_train))} mean:{int(data.rent.mean())}')

print('Test')
print(f'MSE: {mean_squared_error(y_test_orig, pred_test)}')
print(f'MAPE: {mean_absolute_percentage_error(y_test_orig, pred_test)}')
print(f'RSQ: {r2_score(y_test_orig, pred_test)}')
print(f'RMSE: {sqrt(mean_squared_error(y_test_orig, pred_test))} mean:{int(data.rent.mean())}')

plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test_orig, y=pred_test, alpha=0.6, s=50, color="#1f77b4", edgecolor='white', linewidth=0.5)

max_val = max(y_test_orig.max(), pred_test.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1.5)

plt.xlabel("Gerçek", fontsize=12)
plt.ylabel("Tahmin", fontsize=12)
plt.legend(frameon=False)
sns.despine()
plt.grid(False)
plt.tight_layout()
plt.show()


# Linear Regression
data_core = data.copy()

data_core['contact_Contact Owner'] = data_core['contact_Contact Owner'].astype(int)
data_core['area_type_Super Area'] = data_core['area_type_Super Area'].astype(int)
data_core['city_Chennai'] = data_core['city_Chennai'].astype(int)
data_core['city_Delhi'] = data_core['city_Delhi'].astype(int)
data_core['city_Hyderabad'] = data_core['city_Hyderabad'].astype(int)
data_core['city_Kolkata'] = data_core['city_Kolkata'].astype(int)
data_core['city_Mumbai'] = data_core['city_Mumbai'].astype(int)

features = ['size', 'area_type_Super Area', 'bath', 'furnish', 'max_floor', 'floor', 'contact_Contact Owner', 'bhk', 'city_Chennai', 'city_Delhi', 'city_Hyderabad', 'city_Kolkata', 'city_Mumbai']

X = data_core[features]
y = np.log1p(data_core['rent'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_l = LinearRegression()
model_l.fit(X_train, y_train)

pred_train = np.expm1(model_l.predict(X_train))
pred_test = np.expm1(model_l.predict(X_test))
y_train_orig = np.expm1(y_train)
y_test_orig = np.expm1(y_test)

print("Train")
print(f"MSE: {mean_squared_error(y_train_orig, pred_train)}")
print(f"MAPE: {mean_absolute_percentage_error(y_train_orig, pred_train)}")
print(f"RSQ: {r2_score(y_train_orig, pred_train)}")
print(f"RMSE: {sqrt(mean_squared_error(y_train_orig, pred_train))} mean:{int(data_core.rent.mean())}")

print("\nTest")
print(f"MSE: {mean_squared_error(y_test_orig, pred_test)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test_orig, pred_test)}")
print(f"RSQ: {r2_score(y_test_orig, pred_test)}")
print(f"RMSE: {sqrt(mean_squared_error(y_test_orig, pred_test))} mean:{int(data_core.rent.mean())}")

plt.figure(figsize=(6, 4))

sns.scatterplot(x=y_test_orig, y=pred_test - y_test_orig, alpha=0.6, s=50, color="#1f77b4", edgecolor='white', linewidth=0.5)
plt.axhline(y=0, color='green', linestyle='--', linewidth=1.5, label='Zero Error Line')

plt.xlabel("y_test", fontsize=12)
plt.ylabel("y_pred - y_test", fontsize=12)
plt.title("Hatalar vs Gerçek Kira", fontsize=14, weight='bold')
plt.legend(frameon=False)
sns.despine()
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test_orig, y=pred_test, alpha=0.6, s=50, color="#1f77b4", edgecolor='white', linewidth=0.5)

max_val = max(y_test_orig.max(), pred_test.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1.5, label='Perfect (y = x)')

plt.xlabel("y_test", fontsize=12)
plt.ylabel("y_pred", fontsize=12)
plt.title("Tahmin vs Gerçek", fontsize=14, weight='bold')
plt.legend(frameon=False)
sns.despine()
plt.grid(False)
plt.tight_layout()
plt.show()

