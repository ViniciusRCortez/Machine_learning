from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc


dbtrain = pd.read_csv(r'C:\Users\vinis\Downloads\train.csv')
dbtest = pd.read_csv(r'C:\Users\vinis\Downloads\eval.csv')

'''sns.pairplot(dbtrain)
plt.show()
sns.heatmap(dbtrain.corr(), cmap='Wistia', annot=True)
plt.show()

so pega as colunas numericas......'''

y_train = dbtrain.pop('survived')   #pop tira essa coluna do db e salva na variavel em questão
y_test = dbtest.pop('survived')

pd.concat([dbtrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survived')
plt.show()#porcentagem de sobreviventes por sexo
print(dbtrain.info())
print(y_train)
