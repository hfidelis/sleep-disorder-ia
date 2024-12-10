import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

dados = pd.read_csv('../Sleep_health_and_lifestyle_dataset.csv')

print(dados)

dados.rename(
    columns={
        'BMI Category': 'cat_BMI',
        'Person ID':'Person_id',
        'Sleep Duration':'Sleep_duration',
        'Blood Pressure':'Blood_Pressure',
        'Sleep Disorder':'Sleep_Disorder',
    },
    inplace=True,
)

# Transformando os dados para números inteiros

le = LabelEncoder()

le.fit(dados.Gender)
dados.Gender = le.transform(dados.Gender)

le.fit(dados.Occupation)
dados.Occupation = le.transform(dados.Occupation)

le.fit(dados.cat_BMI)
dados.cat_BMI = le.transform(dados.cat_BMI)

le.fit(dados.Blood_Pressure)
dados.Blood_Pressure = le.transform(dados.Blood_Pressure)

le.fit(dados.Sleep_Disorder)
dados.Sleep_Disorder = le.transform(dados.Sleep_Disorder)

# Dataframes com os valores convertidos pelo LabelEconder

def create_label_mapping_df(dados, column_name):
    le = LabelEncoder()

    le.fit(dados[column_name])

    mapping_dict = dict(zip(le.classes_, le.transform(le.classes_)))

    mapping_df = pd.DataFrame({
        'Original_Value': mapping_dict.keys(),
        'Label_Encoded': mapping_dict.values(),
    })

    return mapping_df

labels = [
    'Gender',
    'Occupation',
    'cat_BMI',
    'Blood_Pressure',
    'Sleep_Disorder'
]

for label in labels:
    mapping_df = create_label_mapping_df(dados, label)

    print(f'Mapeamento: {label}')

    print(mapping_df)

# Extrai caracteristicas (variaveis independentes) e variavel alvo

def dados_para_treino(dados):
    label = 'Sleep_Disorder'

    x = dados.drop(label, axis=1)  # Caracteristicas
    y = dados[label]        # Alvo

    return x, y

x, y = dados_para_treino(dados)

# Dividindo os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
  'fit_intercept': [True, False],
  'n_jobs': [-1, 1]  # Use -1 for all CPU cores (adjust if needed)
}

model_lr = LinearRegression()
model_lr = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')

model_ridge = Ridge(alpha=0.1)
model_lasso = Lasso(alpha=0.1)

# Treinando os modelos

model_lr.fit(X_train, y_train)
best_model_lr = model_lr if hasattr(model_lr, 'best_estimator_') else model_lr

model_ridge.fit(X_train, y_train)
best_model_ridge = model_ridge if hasattr(model_ridge, 'best_estimator_') else model_ridge

model_lasso.fit(X_train, y_train)
best_model_lasse = model_lasso if hasattr(model_lasso, 'best_estimator_') else model_lasso

# Fazendo previsões

y_pred_lr = model_lr.predict(X_test)
y_pred_ridge = model_ridge.predict(X_test)
y_pred_lasso = model_lasso.predict(X_test)

# Calculando o MSE

mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Calculando o Coeficiente de Determinacao para medir qualidade da aplicacao

y_pred = best_model_lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Coeficiente de Determinacao (R-squared):", r2)

y_pred = best_model_ridge.predict(X_test)
r2 = r2_score(y_test,y_pred)
print("Coeficiente de Determinacao (R-squared):", r2)

y_pred = best_model_lasse.predict(X_test)
r2 = r2_score(y_test,y_pred)
print("Coeficiente de Determinacao (R-squared):", r2)


print("Regressão Linear MSE:", mse_lr)
print("Regressão Ridge MSE:", mse_ridge)
print("Regressão Lasso MSE:", mse_lasso)

def linear_regression(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    param_grid = {
        'fit_intercept': [True, False],
        'n_jobs': [-1, 1]  # Use -1 for all CPU cores (adjust if needed)
    }

    model_lr = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')
    model_lr.fit(x_train, y_train)

    best_model_lr = model_lr if hasattr(model_lr, 'best_estimator_') else model_lr

    y_pred = best_model_lr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print("(MSE):", mse)
    print("Coeficiente de Determinação (R-squared):", r2)

    if hasattr(model_lr, 'best_params_'):
        print("Melhores hiperparâmetros:", model_lr.best_params_)

def RandomForest(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(x,y)

    y_pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    importances = rf.feature_importances_

    print("MSE: ", mse)
    print("Importancia: ", importances)

x,y = dados_para_treino(dados)

linear_regression(x,y)
RandomForest(x,y)

# Modelagens

# 1. Decision Tree

# Dados treino e teste
x, y = dados_para_treino(dados)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
decisiontree_accuracy = accuracy_score(y_test, y_pred)
print("Acurácia Desision Tree:", decisiontree_accuracy)

# 2. Gradient Boosting

# Dividindo os dados em treino e teste
x, y = dados_para_treino(dados)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Criando e treinando o modelo
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
gradientboosting_accuracy = accuracy_score(y_test, y_pred)
print("Acurácia GradientBoosting:", gradientboosting_accuracy)

# 3. CatBoost

# Dados treino e teste
x, y = dados_para_treino(dados)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    cat_features=None
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
catboost_accuracy = accuracy_score(y_test, y_pred)
print("Acurácia CatBoost:", catboost_accuracy)

# Comparação dos modelos

models = [
    'Decision Tree',
    'Gradient Boosting',
    'CatBoost',
]

accuracies = [
    decisiontree_accuracy,
    gradientboosting_accuracy,
    catboost_accuracy,
]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.ylabel('Acurácia')
plt.xlabel('Modelo')
plt.title('Comparação de Acurácia dos Modelos')
plt.ylim(0, 1)
plt.show()