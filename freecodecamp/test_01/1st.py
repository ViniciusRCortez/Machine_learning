from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc


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
print(x_train.info())
print(y_train)

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
print(feature_list)
