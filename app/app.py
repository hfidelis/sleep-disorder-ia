import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def get_data():
    return pd.read_csv('../Sleep_health_and_lifestyle_dataset.csv')

def train_model():
    data = get_data()

    data.rename(
        columns={
            'BMI Category': 'cat_BMI',
            'Person ID': 'Person_id',
            'Sleep Duration': 'Sleep_duration',
            'Blood Pressure': 'Blood_Pressure',
            'Sleep Disorder': 'Sleep_Disorder',
        },
        inplace=True,
    )

    defaultcols = ['Gender', 'Occupation', 'cat_BMI', 'Blood_Pressure', 'Sleep_Disorder']
    data = data[defaultcols]

    le = LabelEncoder()
    for col in defaultcols:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])

    label = 'Sleep_Disorder'
    x = data.drop(label, axis=1)
    y = data[label]

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(x, y)

    return model

data = get_data()
model = train_model()

st.title("Prevendo Distúrbios do Sono")

st.markdown("Esta é uma aplicação para realizar a predição de distúrbios do sono utilizando o modelo Gradient Boosting Classifier.")

defaultcols = [
    'Gender',
    'Occupation',
    'BMI Category',
    'Blood Pressure',
]

cols = st.multiselect('Atributos', data.columns.tolist(), default=defaultcols)

st.dataframe(data[cols].head(10))

st.sidebar.subheader('Defina os atributos para predição')

gender = st.sidebar.selectbox('Sexo', ('Masculino', 'Feminino'))
occupation = st.sidebar.selectbox('Ocupação', ('Contador', 'Doutor', 'Engenheiro', 'Advogado', 'Gerente', 'Enfermeiro', 'Representante de Vendas', 'Vendedor', 'Cientista', 'Engenheiro de Software', 'Professor'))
imc = st.sidebar.selectbox('IMC', ('Abaixo do Peso', 'Peso Normal', 'Obeso', 'Sobrepreso'))
blood_pressure = st.sidebar.selectbox('Pressão Sanguínea', ('115/75', '115/78', '117/76', '118/75', '118/76', '119/77', '120/80', '121/79', '122/80', '125/80', '125/82', '126/83', '128/84', '128/85', '129/84', '130/85', '130/86', '131/86', '132/87', '135/88', '135/90', '139/91', '140/90', '140/95', '142/92'))

gender_lookup = {
    'Masculino': 1,
    'Feminino': 0,
}

occupation_lookup = {
    'Contador': 0,
    'Doutor': 1,
    'Engenheiro': 2,
    'Advogado': 3,
    'Gerente': 4,
    'Enfermeiro': 5,
    'Representante de Vendas': 6,
    'Vendedor': 7,
    'Cientista': 8,
    'Engenheiro de Software': 9,
    'Professor': 10,
}

imc_lookup = {
    'Abaixo do Peso': 0,
    'Peso Normal': 1,
    'Obeso': 2,
    'Sobrepreso': 3,
}

blood_pressure_lookup = {
    '115/75': 0,
    '115/78': 1,
    '117/76': 2,
    '118/75': 3,
    '118/76': 4,
    '119/77': 5,
    '120/80': 6,
    '121/79': 7,
    '122/80': 8,
    '125/80': 9,
    '125/82': 10,
    '126/83': 11,
    '128/84': 12,
    '128/85': 13,
    '129/84': 14,
    '130/85': 15,
    '130/86': 16,
    '131/86': 17,
    '132/87': 18,
    '135/88': 19,
    '135/90': 20,
    '139/91': 21,
    '140/90': 22,
    '140/95': 23,
    '142/92': 24,
}

gender = gender_lookup[gender]
occupation = occupation_lookup[occupation]
imc = imc_lookup[imc]
blood_pressure = blood_pressure_lookup[blood_pressure]

btn_predict = st.sidebar.button('Realizar predição')

if btn_predict:
    result = model.predict([[gender, occupation, imc, blood_pressure]])
    st.subheader('O provável distúrbio do sono dessa pessoa é: ')
    sleep_disorder_lookup = {
        0: 'Insônia',
        1: 'Apneia do Sono',
        2: 'Nenhum',
    }
    st.write(sleep_disorder_lookup[result[0]])