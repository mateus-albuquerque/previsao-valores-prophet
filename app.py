import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go

# Adiciona a logo do aplicativo
st.image('https://raw.githubusercontent.com/mateus-albuquerque/previsao-valores-prophet/main/logo.png', width=200)

st.title('Previsão de Valores Futuros')
uploaded_file = st.file_uploader("Carregue sua planilha Excel", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    df['Data Emissão'] = pd.to_datetime(df['Data Emissão'], errors='coerce')
    df['Valor Liquido Documento'] = pd.to_numeric(df['Valor Liquido Documento'], errors='coerce')
    
    df.dropna(subset=['Data Emissão', 'Valor Liquido Documento'], inplace=True)
    df = df[df['Valor Liquido Documento'] != 0]
    df.set_index('Data Emissão', inplace=True)
    
    df['Ano'] = df.index.year
    df['Mês'] = df.index.month
    df['Dia da Semana'] = df.index.dayofweek
    df['Dia do Mês'] = df.index.day
    
    df_prophet = df[['Valor Liquido Documento']].reset_index()
    df_prophet.rename(columns={'Data Emissão': 'ds', 'Valor Liquido Documento': 'y'}, inplace=True)
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df_prophet)

    future = model_prophet.make_future_dataframe(periods=6)
    forecast = model_prophet.predict(future)

    forecast_6d = forecast[['ds', 'yhat']].tail(6)
    forecast_6d.set_index('ds', inplace=True)

    y_true = df['Valor Liquido Documento'].reindex(forecast_6d.index)

    valid_indices = y_true.notna()
    y_true = y_true[valid_indices]
    forecast_values_aligned = forecast_6d.loc[valid_indices, 'yhat']

    last_15_days = df.tail(15)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=last_15_days.index,
        y=last_15_days['Valor Liquido Documento'],
        mode='lines+markers',
        name='Dados Reais',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_6d.index,
        y=forecast_6d['yhat'],
        mode='lines+markers',
        name='Previsão 5 Dias',
        line=dict(color='red', dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Previsão de Valores Futuros - Últimos 15 Dias',
        xaxis_title='Data',
        yaxis_title='Valor Líquido Documento',
        legend_title='Legenda',
        template='plotly_white'
    )

    st.plotly_chart(fig)

    st.subheader('Tabela de Previsão para os Próximos 6 Dias')
    forecast_df_formatted = pd.DataFrame({
        'Data': forecast_6d.index,
        'Previsão': ['R$ {:,.2f}'.format(val).replace(',', 'X').replace('.', ',').replace('X', '.') for val in forecast_6d['yhat']]
    })
    st.write(forecast_df_formatted)

else:
    st.write("Por favor, carregue uma planilha Excel para iniciar a análise.")
