import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Carregar os dados da planilha em um DataFrame
df = pd.read_excel("dados_turismo.xlsx")


# Lista para armazenar os resultados de cada mês
resultados = []

# Loop para iterar por cada mês
for mes in range(1, 13):  # de janeiro a dezembro
    # Filtrar os dados apenas para o mês atual
    df_mes = df[df['Mes'] == mes]

    # Separa os dados de entrada (variáveis independentes) e saída (gasto médio e injeção financeira)
    features = ['Taxa_Ocupacao_Hospedagem', 'Tempo_Medio_Permanencia', 'Percentual_Meios_Hospedagem',
                'Percentual_Excursionistas', 'Outras_Formas_Hospedagem', 'Numero_UHs_Janeiro_2016', 'Total_Visitantes']
    X = df_mes[features]
    y_gasto_medio = df_mes['Gasto_Medio']
    y_injecao_financeira = df_mes['Injecao_Financeira']

    # Cria os modelos de regressão de Árvore de Decisão para o Gasto Médio e Injeção Financeira
    modelo_gasto_medio = DecisionTreeRegressor()
    modelo_gasto_medio.fit(X, y_gasto_medio)

    modelo_injecao_financeira = DecisionTreeRegressor()
    modelo_injecao_financeira.fit(X, y_injecao_financeira)

    # Previsões para o mês atual
    gasto_medio_previsto = modelo_gasto_medio.predict(X)
    injecao_financeira_prevista = modelo_injecao_financeira.predict(X)

    # Armazenar os resultados do mês atual na lista
    resultados.append({
        'Mes': mes,
        'Gasto_Medio_Real': y_gasto_medio.values,
        'Gasto_Medio_Previsto': gasto_medio_previsto,
        'Injecao_Financeira_Real': y_injecao_financeira.values,
        'Injecao_Financeira_Prevista': injecao_financeira_prevista
    })

# Criar um gráfico de dispersão para comparar as previsões com os valores reais para Gasto Médio
plt.figure(figsize=(10, 8))
for resultado in resultados:
    mes = resultado['Mes']
    plt.scatter(resultado['Gasto_Medio_Real'], resultado['Gasto_Medio_Previsto'], label=f'Gasto Médio - Mês {mes}')

plt.xlabel('Real')
plt.ylabel('Previsto')
plt.title('Comparação entre Gasto Médio Real e Previsto')
plt.legend()
plt.grid(True)
plt.show()

# Criar um gráfico de dispersão para comparar as previsões com os valores reais para Injeção Financeira
plt.figure(figsize=(10, 8))
for resultado in resultados:
    mes = resultado['Mes']
    plt.scatter(resultado['Injecao_Financeira_Real'], resultado['Injecao_Financeira_Prevista'], label=f'Injeção Financeira - Mês {mes}')

plt.xlabel('Real')
plt.ylabel('Previsto')
plt.title('Comparação entre Injeção Financeira Real e Prevista')
plt.legend()
plt.grid(True)
plt.show()
