from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
import os


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    #epochs(quantas vezes verei o codigo)/batch(tamanho do foton de dados)/data(x)/label(y)
    def input_function():   #tranforma o retorno em uma função
        ds = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))#preparando o tensor treino
        if shuffle:
            ds = ds.shuffle(1000)#embaralha os dados
        ds = ds.batch(batch_size).repeat(num_epochs)#treina varias vezes
        return ds
    return input_function


x_train = pd.read_csv(r'C:\Users\vinis\Downloads\train.csv')
x_test = pd.read_csv(r'C:\Users\vinis\Downloads\eval.csv')

'''sns.pairplot(dbtrain)
plt.show()
sns.heatmap(dbtrain.corr(), cmap='Wistia', annot=True)
plt.show()

so pega as colunas numericas......'''

y_train = x_train.pop('survived')   #pop tira essa coluna do db e salva na variavel em questão
y_test = x_test.pop('survived')

pd.concat([x_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survived')
#plt.show()#porcentagem de sobreviventes por sexo

#Transformando categorias em numeros:
feature_list = []
categorical_column = ['sex', 'class', 'deck', 'embark_town', 'alone']
numeric_column = ['age', 'fare', 'n_siblings_spouses', 'parch']
for item in categorical_column:
     vocabulary = x_train[item].unique()
     feature_list.append(tf.feature_column.categorical_column_with_vocabulary_list(item, vocabulary))
     #gera uma lista com os valores unicos de cada item e add eles
for item in numeric_column:
    feature_list.append(tf.feature_column.numeric_column(item, dtype=tf.float32))#add os valores numaricos


#Treinando modelo:
train_input_data = make_input_fn(x_train, y_train)
test_input_data = make_input_fn(x_test, y_test, num_epochs=1, shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_list)
linear_est.train(train_input_data) #treina
result = linear_est.evaluate(test_input_data)#testa
os.system('cls')
print(result)    
#Para prever:
result = linear_est.predict('POR AQUI UMA LINHA INTEIRA DE DADOS')
