#!/usr/bin/env python
# coding: utf-8

# ## Prédiction de prix de vente de maisons

# In[1]:


# Importer les bibliotheques 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import KFold

# Contrôle l'affichage des colonnes
#https://towardsdatascience.com/8-commonly-used-pandas-display-options-you-should-know-a832365efa95
pd.set_option('display.max_columns', 500)


# In[2]:


# importer les données 
data = pd.read_csv("AmesHousing.tsv", delimiter='\t')


# In[3]:


#Afficher la forme de la table 
print(data.shape)
print(len(str(data.shape))*'-')

#Afficher les totaux de diférents types de colonnes 
print(data.dtypes.value_counts())

#Afficher la table 
data.head()


# In[4]:


# Fonctions permettant de calculer l'erreur quadratique moyenne de la regression de linéaire 
# Plus l'erreur est faible plus le modéle est proche de la réalité
def transform_features(df):
    return df

def select_features(df):
    return df[["Gr Liv Area", "SalePrice"]]

def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    numeric_train = train.select_dtypes(include=['integer', 'float'])
    numeric_test = test.select_dtypes(include=['integer', 'float'])
    features = numeric_train.columns.drop("SalePrice")
    
    # Entrainement
    lr = linear_model.LinearRegression()
    lr.fit(train[features], train["SalePrice"])
    
    # Prédiction
    predictions = lr.predict(test[features])
    mse = mean_squared_error(test["SalePrice"], predictions)
    rmse = np.sqrt(mse)
    
    return rmse


# In[5]:


#Application de 
transform_df = transform_features(data)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)
rmse


# On reamrque que sans faire de prétraitement au préalable de nos caractéristiques, de la sélection des caractéristiques, l'erreur quadratique moyenne (rmse) est très grande. Nous allons faire un peu de traitement pour voir s'il y a un changment de valeur de rmse. Il faut noter que le modéle de régression linéaire est sensible au probléme de dimension, appelée communement la malédiction de la dimension. 

# ## Traitement des caractéristiques
# 
# Le succés de la modélisation predictive depend fortement des qualités des caracteristiques du modéle 

# Gérer les valeurs manquantes:
# - Toutes les colonnes:
#  + Supprimer celles qui ont 5% ou plus de valeurs manquantes pour le moment.
# - Colonnes texte:
#  + Supprimer toute colonne contenant 1 valeur manquante ou plus pour le moment.
# - Colonnes numériques:
#  + Pour les colonnes contenant des valeurs manquantes, remplacer par les valeurs les plus fréquentes de la colonne
#  
# **B :** il faut absolument être prudent avec les valeurs manquantes leur totale supression peut causer la perte d'informations qui pourraient être utiles. Donc, on doit trouver des combines pour optimiser leur gestion et ainsi améliorer les performances du modéle.

# **1. Toutes les colonnes:** supprimer celles qui ont 5% ou plus de valeurs manquantes pour le moment.

# In[6]:


# Nombre de variables manquantes
num_missing = data.isnull().sum()

# Filtrer l'objet Series sur les colonnes contenant > 5% de valeurs manquantes
drop_missing_cols = num_missing[(num_missing > len(data)/20)].sort_values()

# Supprimer ces colonnes dans le DataFrame. Noter l'utilisation de l'accesseur .index
data = data.drop(drop_missing_cols.index, axis=1)
data.shape[1]


# On constate que des 82 colonnes, on est passé à 71 en supprimant toutes les colonnes ayant plus de 5% de valeurs manquantes. On continue le pretraitement de colonnes texte dont on appelle communement des variables catégorielles. 

# **2. Colonnes texte :** supprimer toute colonne contenant au moins une valeur manquante pour le moment.

# In[7]:


text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(
    ascending=False)

# Filtrer l'objet Series sur les colonnes contenant au moins une valeur manquante
drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
data = data.drop(drop_missing_cols_2.index, axis=1)
data.shape[1]


# On constate qu'on a passé de 71 colonnes à 64 en supprimant toutes les variables catégorielles ayent au moins une valeur manquante 

# **3. Colonnes numériques :** pour les colonnes contenant des valeurs manquantes, remplacer par la valeur la plus fréquente de la colonne

# In[12]:


# Calculer le nombre de valeurs manquantes colonne par colonne
num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()

# récuperer et trier le 5% de valeurs manquantes des colonnes numériques 
fixable_numeric_missing_cols = num_missing[(num_missing < len(data)/20) & 
                                   (num_missing > 0)].sort_values()
fixable_numeric_missing_cols


# In[23]:


# Calculer la valeur la plus commune pour chaque colonne.
replacement_values_dict = data[fixable_numeric_missing_cols.index].mode().to_dict(
    orient='records')[0]
replacement_values_dict


# In[11]:


# Remplacer les valeurs manquantes.
data = data.fillna(replacement_values_dict)

## Vérifier que toutes les colonnes ont bien 0 valeur manquante
data.isnull().sum().value_counts()


# Nous avons fini avec les valeurs manquantes. Nous allos continuer à faire de feature engineering en créant des nouvelles caractéristiques

# **Quelles nouvelles caractéristiques pouvons-nous créer pour mieux capturer les informations contenues dans certaines caractéristiques?**

# In[30]:


# Les dates de vente moins les dates de construction 
years_sold = data['Yr Sold'] - data['Year Built']
years_sold[years_sold < 0]


# Nous avons une maison qui a été vendue 1 an avant sa constrcution 

# In[29]:


# Comparons les dates de ventes et de renovation
years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
years_since_remod[years_since_remod < 0]


# In[14]:


# Créer 2 nouvelles colonnes
data['Years Before Sale'] = years_sold
data['Years Since Remod'] = years_since_remod


# In[15]:


# Supprimer les lignes avec des valeurs négatives pour ces nouvelles caractéristiques
data = data.drop([1702, 2180, 2181], axis=0)


# In[16]:


# Plus besoin des colonnes de l'année d'origine
data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)


# Supprimer les colonnes qui:
# - Ne sont pas utiles pour le Machine Learning
# - Fuite des données au sujet de la vente finale (cf. documentation dataset)

# In[24]:


# Supprimer les colonnes qui ne sont pas utiles pour le ML
data = data.drop(["PID", "Order"], axis=1)

# Supprimer les colonnes qui font fuiter des informations sur la vente finale
data = data.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)


# In[25]:


data.head()


# In[31]:


# Une fonction contenant tous les pretraitements des caractéristiques rélisés ci-haut

def transform_features(df, percent_missing=0.05):
    num_missing = df.isnull().sum()
    drop_missing_cols = num_missing[(num_missing > len(df)*percent_missing)].sort_values()
    df = df.drop(drop_missing_cols.index, axis=1)
    
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(
        ascending=False)
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    df = df.drop(drop_missing_cols_2.index, axis=1)
    
    num_missing = df.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(df)*percent_missing) & 
                                       (num_missing > 0)].sort_values()
    replacement_values_dict = df[fixable_numeric_cols.index].mode().to_dict(
        orient='records')[0]
    df = df.fillna(replacement_values_dict)
    
    years_sold = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    df['Years Before Sale'] = years_sold
    df['Years Since Remod'] = years_since_remod
    df = df.drop([1702, 2180, 2181], axis=0)
    df = df.drop(["PID", "Order", "Mo Sold", "Sale Condition", "Sale Type", "Year Built", 
                  "Year Remod/Add"], axis=1)
    
    return df

#Fonction retournant deux caractéristiques
def select_features(df):
    return df[["Gr Liv Area", "SalePrice"]]

#Fonction rettournant la valeur de rmse 
def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    numeric_train = train.select_dtypes(include=['integer', 'float'])
    numeric_test = test.select_dtypes(include=['integer', 'float'])
    features = numeric_train.columns.drop("SalePrice")
    
    # Entrainement
    lr = linear_model.LinearRegression()
    lr.fit(train[features], train["SalePrice"])
    
    # Prédiction
    predictions = lr.predict(test[features])
    mse = mean_squared_error(test["SalePrice"], predictions)
    rmse = np.sqrt(mse)
    
    return rmse


# In[32]:


data = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(data)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)
rmse


# Après avoir traité et retravaillé nos caractéristiques, on obtient une autre valeur d'erreur de 55275.367. Donc, on peut dire on a amélioré notre modéle simplement en transformant nos caractéristiques

# ## Sélection des caractéristiques

# In[34]:


# Selectinner les colonnes numériques
numerical_df = transform_df.select_dtypes(include=['int', 'float'])
numerical_df.head()


# In[35]:


# Estimer la correlation de la colonne cible (SalePrice) avec les différentes caracteristiques 
abs_corr_coeffs = numerical_df.corr()['SalePrice'].abs().sort_values()
abs_corr_coeffs


# In[36]:


# Ne gardons que les colonnes avec un coefficient de corrélation supérieur à 0.4 
# (arbitraire, à tester plus tard!)
abs_corr_coeffs[abs_corr_coeffs > 0.4]


# In[37]:


# Supprimer les colonnes avec une corrélation inférieure à 0.4 avec SalePrice
transform_df = transform_df.drop(abs_corr_coeffs[abs_corr_coeffs < 0.4].index, axis=1)
transform_df.shape[1]


# Quelles colonnes catégoriques devrions-nous garder?

# In[39]:


# Créer une liste de noms de colonne à partir de la documentation qui sont censés être 
# catégoriques
nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley",
                    "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", 
                    "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd", 
                    "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type",
                    "Misc Feature", "Sale Type", "Sale Condition"]


# - Quelles colonnes sont actuellement numériques mais doivent plutôt être codées en tant que catégoriques (car les nombres n'ont aucun signification sémantique)?
# - Si une colonne catégorique contient des centaines de valeurs uniques (ou catégories), devrions-nous la conserver? Lorsque nous rendons cette colonne factice, des centaines de colonnes (pour chacune des catégories) devront être rajoutées au DataFrame.

# In[40]:


transform_cat_cols = []
for col in nominal_features:
    if col in transform_df.columns:
        transform_cat_cols.append(col)
transform_cat_cols


# In[41]:


# Combien de valeurs uniques dans chaque colonne catégorique?
uniqueness_counts = transform_df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()

# Limite arbitraire de 10 valeurs uniques (expérimentation)
drop_nonuniq_cols = uniqueness_counts[uniqueness_counts > 10].index
transform_df = transform_df.drop(drop_nonuniq_cols, axis=1)


# In[42]:


# Sélectionner uniquement les colonnes de texte restantes et convertissez-les en catégories
text_cols = transform_df.select_dtypes(include=['object'])
for col in text_cols:
    transform_df[col] = transform_df[col].astype('category')


# In[43]:


# Créer des colonnes factices et ajouter les au DataFrame
transform_df = pd.concat([
    transform_df,
    pd.get_dummies(transform_df.select_dtypes(include=['category']))
], axis=1)


# In[44]:


transform_df.head()


# In[49]:


# Fonction realisant les étapes de la sélection des caractéristiuqes 
def select_features(df, coeff_threshold=0.4, uniq_threshold=10):
    numerical_df = df.select_dtypes(include=['int', 'float'])
    abs_corr_coeffs = numerical_df.corr()['SalePrice'].abs().sort_values()
    df = df.drop(abs_corr_coeffs[abs_corr_coeffs < coeff_threshold].index, axis=1)
    
    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour",
                        "Lot Config", "Neighborhood", "Condition 1", "Condition 2", 
                        "Bldg Type", "House Style", "Roof Style", "Roof Matl","Exterior 1st",
                        "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", 
                        "Central Air", "Garage Type","Misc Feature", "Sale Type", 
                        "Sale Condition"]
    
    transform_cat_cols = []
    for col in nominal_features:
        if col in df.columns:
            transform_cat_cols.append(col)
            
    uniqueness_counts = df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
    drop_nonuniq_cols = uniqueness_counts[uniqueness_counts > uniq_threshold].index
    df = df.drop(drop_nonuniq_cols, axis=1)
    
    text_cols = df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')
    df = pd.concat([df, pd.get_dummies(df.select_dtypes(include=['category']))], axis=1)
    
    return df


# ## Entrainement et test

# In[50]:


def train_and_test(df, k=0):
    numeric_df = df.select_dtypes(include=['integer', 'float'])
    features = numeric_df.columns.drop("SalePrice")
    lr = linear_model.LinearRegression()
    
    if k == 0:
        train = df[:1460]
        test = df[1460:]
        
        lr.fit(train[features], train["SalePrice"])
        predictions = lr.predict(test[features])
        
        mse = mean_squared_error(test["SalePrice"], predictions)
        rmse = np.sqrt(mse)
        
        return rmse
    
    if k == 1:
        shuffled_df = df.sample(frac=1, )
        train = df[:1460]
        test = df[1460:]
        
        lr.fit(train[features], train["SalePrice"])
        predictions_one = lr.predict(test[features])
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        lr.fit(test[features], test["SalePrice"])
        predictions_two = lr.predict(train[features])
        
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        
        avg_rmse = np.mean([rmse_one, rmse_two])
        print(rmse_one)
        print(rmse_two)
        
        return avg_rmse
    
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        
        return avg_rmse


# In[ ]:


data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data)
final_data = select_features(transformed_data)

results = []
for i in range(100):
    result = train_and_test(final_data, k=i)
    results.append(result)
    
x = [i for i in range(100)]
y = results 
plt.plot(x, y)
plt.xlabel('Kfolds')
plt.ylabel('RMSE')


# Au final l'erreur le plus faible est atteinte pour le K le plus grand. Bref, la validation croisée nous a permis d'évaluer la performance du mondéle on quasiment la rmse en deux avec k avoisiant le 100

# In[1]:


get_ipython().system('export PATH=/Library/TeX/texbin:$PATH')


# In[ ]:




