import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/home/guilherme/Documentos/meus-codigos/efeitos_adversos/efeito_adverso_de_remedios.csv")
novos_pacientes = pd.read_csv("/home/guilherme/Documentos/meus-codigos/efeitos_adversos/novos_pacientes.csv")

# Objetivos:
# - Ver quais foram os grupos de pessoas mais afetados por efeitos adversos
# - Conseguir prever se novos pacientes terão esses efeitos

# Observações:
# - idade, peso e comorbidade são as minhas variáveis preditoras
# - efeito_adverso é a minha variável alvo

# Criando o dataset `datasetAfetados`
datasetAfetados = dataset.loc[dataset["efeito_adverso"]=="s"]

# Média de idade e peso dos afetados
print("Média de idade dos afetados (anos): {0:.1f} \n".format(datasetAfetados["idade"].mean()))
print("Média de peso dos afetados (quilogramas): {0:.1f} \n".format(datasetAfetados["peso"].mean()))

# Boxplot (diagrama de caixa) de idade e peso dos afetados
print("Mostrando boxplot (diagrama de caixa) de idade e peso dos afetados... \n")
datasetAfetados.boxplot(column = ["idade","peso"])
plt.show()

# Gráfico de dispersão das idades e pesos dos afetados (cada ponto equivale a uma pessoa)
# Isso mostra o grupo de idade e peso mais afetados
print("Mostrando gráfico de dispersão da idade e do peso dos afetados... \n")
print("Segundo o gráfico de dispersão, a faixa etária mais afetada foi de 40 a 60 anos e a faixa de peso foi de 0 a 50 quilogramas \n")
plt.scatter(datasetAfetados["idade"],datasetAfetados["peso"])
plt.show()




# Porcentagem de afetados e não afetados
afetados_porcentagem = 13*100/30
nao_afetados_porcentagem = 17*100/30
print("{0:.0f}% das pessoas foram afetadas e {1:.0f}% não foram afetadas (números aproximados)".format(afetados_porcentagem,nao_afetados_porcentagem))


# Criando um modelo para ver se novos pacientes teram ou não efeitos adversos
# sim = s = 1 e não = n = 0


# Criando uma cópia do dataset para fazer o modelo
datasetPrevisao = dataset

# Trocando 's' e 'n' por números
datasetPrevisao["comorbidade"] = datasetPrevisao["comorbidade"].replace("s",1)
datasetPrevisao["comorbidade"] = datasetPrevisao["comorbidade"].replace("n",0)

datasetPrevisao["efeito_adverso"] = datasetPrevisao["efeito_adverso"].replace("s",1)
datasetPrevisao["efeito_adverso"] = datasetPrevisao["efeito_adverso"].replace("n",0)
#print(datasetPrevisao)

# Separando em variáveis preditoras e variáveis alvo
variavel_alvo = datasetPrevisao["efeito_adverso"]
variaveis_preditoras = datasetPrevisao.drop("efeito_adverso", axis = 1)
variaveis_preditoras.drop("paciente", axis = 1, inplace = True)

#print(variavel_alvo.head())
#print(variaveis_preditoras.head())

# Separando em treino e teste
from sklearn.model_selection import train_test_split
variaveis_preditoras_treino, variaveis_preditoras_teste, variavel_alvo_treino, variavel_alvo_teste = train_test_split(variaveis_preditoras, variavel_alvo, test_size = 0.3)

# A quantidade de linhas dos datasets de treino (variaveis_preditoras_treino e variavel_alvo_treino) devem ser iguais, o mesmo vale para os de teste (variaveis_preditoras_teste e variavel_alvo_teste)
#print(variaveis_preditoras_treino.shape)
#print(variavel_alvo_treino.shape)
#print(variaveis_preditoras_teste.shape)
#print(variavel_alvo_teste.shape)

# Importando o algoritmo
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()

# Treinando-o
modelo.fit(variaveis_preditoras_treino, variavel_alvo_treino)

# Testando-o
resultado = modelo.score(variaveis_preditoras_teste,variavel_alvo_teste)

# E mostrando sua acurácia em decimal
print("Acurácia do modelo: {0}% \n".format(resultado*100))

# Prevendo valores com novos pacientes
novos_pacientesPrevisao = novos_pacientes

novos_pacientesPrevisao["comorbidade"] = novos_pacientesPrevisao["comorbidade"].replace("s",1)
novos_pacientesPrevisao["comorbidade"] = novos_pacientesPrevisao["comorbidade"].replace("n",0)
novos_pacientesPrevisao.drop("paciente", axis = 1, inplace = True)
novos_pacientesPrevisao.drop("efeito_adverso", axis = 1, inplace = True)
#print(novos_pacientesPrevisao.head())

resultado = modelo.predict(novos_pacientesPrevisao)

# Convertendo 0 e 1 em 'n' e 's'
resultado_lista = []
for e in resultado:
    if e == 1:
        resultado_lista.append("s")
    else:
        resultado_lista.append("n")

print("Os resultados da previsão de efeitos adversos em 5 novos pacientes foram:", resultado_lista, "\n")