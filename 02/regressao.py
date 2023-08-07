import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar os dados da planilha em um DataFrame
df = pd.read_excel("dados_turismo.xlsx")

# Lista para armazenar os resultados de cada mês
resultados = []

# Loop para iterar por cada mês
for mes in range(1, 13):  # de janeiro a dezembro
    # Filtrar os dados apenas para o mês atual
    df_mes = df[df['Mes'] == mes]

    # Verificar se há dados suficientes para dividir em treinamento e teste
    if len(df_mes) < 2:
        print(f"O mês {mes} possui menos de dois exemplos. Não é possível fazer a divisão.")
        continue

    # Separa os dados de entrada (variáveis independentes) e saída (gasto médio e injeção financeira)
    features = ['Taxa_Ocupacao_Hospedagem', 'Tempo_Medio_Permanencia', 'Percentual_Meios_Hospedagem',
                'Percentual_Excursionistas', 'Outras_Formas_Hospedagem', 'Numero_UHs_Janeiro_2016', 'Total_Visitantes']
    X = df_mes[features]
    y_gasto_medio = df_mes['Gasto_Medio']
    y_injecao_financeira = df_mes['Injecao_Financeira']

    # Dividir os dados em conjuntos de treinamento e teste (80% treinamento, 20% teste)
    X_train, X_test, y_gasto_medio_train, y_gasto_medio_test, y_injecao_financeira_train, y_injecao_financeira_test = train_test_split(
        X, y_gasto_medio, y_injecao_financeira, test_size=0.2, random_state=42)

    # Cria os modelos de regressão de Árvore de Decisão para o Gasto Médio e Injeção Financeira
    modelo_gasto_medio = DecisionTreeRegressor()
    modelo_gasto_medio.fit(X_train, y_gasto_medio_train)

    modelo_injecao_financeira = DecisionTreeRegressor()
    modelo_injecao_financeira.fit(X_train, y_injecao_financeira_train)

    # Previsões para o conjunto de teste
    gasto_medio_previsto = modelo_gasto_medio.predict(X_test)
    injecao_financeira_prevista = modelo_injecao_financeira.predict(X_test)

    # Armazenar os resultados do mês atual na lista
    resultados.append({
        'Mes': mes,
        'Gasto_Medio_Real': y_gasto_medio_test.values,
        'Gasto_Medio_Previsto': gasto_medio_previsto,
        'Injecao_Financeira_Real': y_injecao_financeira_test.values,
        'Injecao_Financeira_Prevista': injecao_financeira_prevista
    })

# Gráfico de dispersão para comparar as previsões com os valores reais para Gasto Médio
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

# Gráfico de dispersão para comparar as previsões com os valores reais para Injeção Financeira
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

# Gráfico de Linhas para mostrar a variação do Gasto Médio ao longo dos meses
plt.figure(figsize=(10, 6))
plt.plot(range(1, 13), df.groupby('Mes')['Gasto_Medio'].mean(), marker='o', linestyle='-', color='b')
plt.xlabel('Mês')
plt.ylabel('Gasto Médio')
plt.title('Variação do Gasto Médio ao longo dos meses')
plt.grid(True)
plt.show()

# Gráfico de Barras para comparar os valores reais e previstos do Gasto Médio para cada mês
meses = df['Mes'].unique()
gasto_medio_real = df.groupby('Mes')['Gasto_Medio'].mean()
gasto_medio_previsto = [resultado['Gasto_Medio_Previsto'].mean() for resultado in resultados]

plt.figure(figsize=(10, 6))
plt.bar(meses, gasto_medio_real, width=0.4, label='Real')
plt.bar(meses + 0.4, gasto_medio_previsto, width=0.4, label='Previsto')
plt.xlabel('Mês')
plt.ylabel('Gasto Médio')
plt.title('Comparação entre Gasto Médio Real e Previsto por Mês')
plt.xticks(meses + 0.2, meses)
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de Resíduos para verificar a aleatoriedade dos erros de previsão do modelo
for resultado in resultados:
    mes = resultado['Mes']
    y_gasto_medio_real = resultado['Gasto_Medio_Real']
    y_gasto_medio_previsto = resultado['Gasto_Medio_Previsto']

    residuos_gasto_medio = [y_real - y_previsto for y_real, y_previsto in zip(y_gasto_medio_real, y_gasto_medio_previsto)]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_gasto_medio_real)), residuos_gasto_medio, marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Exemplo')
    plt.ylabel('Resíduos')
    plt.title(f'Gráfico de Resíduos para Gasto Médio - Mês {mes}')
    plt.grid(True)
    plt.show()

# Gráfico de Importância de Recursos por mês
for resultado in resultados:
    mes = resultado['Mes']
    df_mes = df[df['Mes'] == mes]
    X_test = df_mes[features]

    modelo_gasto_medio = DecisionTreeRegressor()
    modelo_gasto_medio.fit(X_train, y_gasto_medio_train)

    importancias = modelo_gasto_medio.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.bar(features, importancias)
    plt.xlabel('Recurso')
    plt.ylabel('Importância')
    plt.title(f'Importância dos Recursos na Previsão do Gasto Médio - Mês {mes}')
    plt.grid(True)
    plt.show()

# Criar um DataFrame para armazenar os novos dados previstos
df_saida = pd.DataFrame()

# Adicionar colunas para o mês e os valores previstos de Gasto Médio e Injeção Financeira
df_saida['Mes'] = range(1, 13)
df_saida['Gasto_Medio_Previsto'] = [resultado['Gasto_Medio_Previsto'].mean() for resultado in resultados]
df_saida['Injecao_Financeira_Prevista'] = [resultado['Injecao_Financeira_Prevista'].mean() for resultado in resultados]

# Exibir a tabela de saída
print(df_saida)

# Salvar a tabela em um arquivo Excel (opcional)
df_saida.to_excel('saida_previsoes.xlsx', index=False)


