import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

excel_file = 'Planilha IA.xlsx'  # Substitua pelo caminho real do arquivo
data = pd.read_excel(excel_file)


# Processamento das informações
data['TENTADO/CONSUMADO'] = data['TENTADO/CONSUMADO'].replace({'TENTADO': 0, 'CONSUMADO': 1})
data['NATUREZA'] = data['NATUREZA'].map(natureza_mapping)

# Dividir os dados em features (X) e target (y)
X = data[['MÊS', 'ANO', 'NATUREZA', 'QUANTIDADE VITIMAS']]
y = data['TENTADO/CONSUMADO']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de Floresta Aleatória
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Validação cruzada para avaliação mais robusta
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Validação Cruzada - Acurácias: {cv_scores}")
print(f"Acurácia Média: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Acurácia: {accuracy:.2f}')
print(f'Relatório de Classificação:\n{classification_rep}')

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Previsão")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.show()
