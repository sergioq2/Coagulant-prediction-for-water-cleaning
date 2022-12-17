rir# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:20:58 2021

@author: squintra
"""
#Importar los datos
#!pip install lightgbm
#!pip install xgboost

#!pip install boruta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import math
!pip install config
import config

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as ltb
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from plotnine import ggplot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
import graphviz
from sklearn import ensemble

from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree

import joblib

"""Exploración de datos"""
"""Analítica exploratoria"""
##Color
color = pd.read_excel('Color.xlsx',sheet_name='Color')
color['Color'].fillna(method='ffill',inplace=True)

#Turbiedad
turbiedad1 = pd.read_excel('Turbiedad.xlsx',sheet_name='Turbiedad_2016')
turbiedad2 = pd.read_excel('Turbiedad.xlsx',sheet_name='Turbiedad_2018')
turbiedad2.drop(['Señal'],axis=1,inplace=True)
turbiedad3 = pd.read_excel('Turbiedad.xlsx',sheet_name='Turbiedad_2020')
turbiedad3.drop(['Señal'],axis=1,inplace=True)

frames_turb = [turbiedad1, turbiedad2, turbiedad3]
datos_turbiedad = pd.concat(frames_turb)

#Conductividad
conductividad1 = pd.read_excel('Conductividad.xlsx',sheet_name='Conductividad_2016')
conductividad1.drop(['Señal'],axis=1,inplace=True)
conductividad2 = pd.read_excel('Conductividad.xlsx',sheet_name='Conductividad_2018')
conductividad2.drop(['Señal'],axis=1,inplace=True)
conductividad3 = pd.read_excel('Conductividad.xlsx',sheet_name='Conductividad_2020')
conductividad3.drop(['Señal'],axis=1,inplace=True)

frames_cond = [conductividad1, conductividad2, conductividad3]
datos_conductividad = pd.concat(frames_cond)

##PH
ph1 = pd.read_excel('PH.xlsx',sheet_name='PH_2016')
ph1.drop(['Señal'],axis=1,inplace=True)
ph2 = pd.read_excel('PH.xlsx',sheet_name='PH_2018')
ph2.drop(['Señal'],axis=1,inplace=True)
ph3 = pd.read_excel('PH.xlsx',sheet_name='PH_2020')
ph3.drop(['Señal'],axis=1,inplace=True)

frames_ph = [ph1, ph2, ph3]
datos_ph = pd.concat(frames_ph)

##Temperatura
temperatura1 = pd.read_excel('Temperatura_siata.xlsx',sheet_name='2016')
temperatura2 = pd.read_excel('Temperatura_siata.xlsx',sheet_name='2017')
temperatura3 = pd.read_excel('Temperatura_siata.xlsx',sheet_name='2018')
temperatura4 = pd.read_excel('Temperatura_siata.xlsx',sheet_name='2019')
temperatura5 = pd.read_excel('Temperatura_siata.xlsx',sheet_name='2020')
temperatura6 = pd.read_excel('Temperatura_siata.xlsx',sheet_name='2021')

frames_temperatura = [temperatura1, temperatura2, temperatura3, temperatura4, temperatura5, temperatura6]
datos_temperatura = pd.concat(frames_temperatura)

##Precipitación
precipitacion1 = pd.read_excel('Precipitacion.xlsx',sheet_name='2016')
precipitacion2 = pd.read_excel('Precipitacion.xlsx',sheet_name='2017')
precipitacion3 = pd.read_excel('Precipitacion.xlsx',sheet_name='2018')
precipitacion4 = pd.read_excel('Precipitacion.xlsx',sheet_name='2019')
precipitacion5 = pd.read_excel('Precipitacion.xlsx',sheet_name='2020')
precipitacion6 = pd.read_excel('Precipitacion.xlsx',sheet_name='2021')

frames_precipitacion = [precipitacion1, precipitacion2, precipitacion3, precipitacion4, precipitacion5, precipitacion6]
datos_precipitacion = pd.concat(frames_precipitacion)

##Dosis
dosis = pd.read_excel('Dosis.xlsx',sheet_name='Dosis')
dosis['Dosis'].fillna(method='ffill',inplace=True)

##Caudales
caudal = pd.read_excel('Caudales.xlsx',sheet_name='caudal')
caudal['Caudal'].fillna(method='ffill',inplace=True)

##Análisis descriptivo inicial
##Limpieza de datos

#Color
color.shape
color.info()
color.isnull().sum()
color['Color'].describe()
plt.boxplot(color['Color'])

color['Color'].quantile(0.95)
color['Color'][color['Color']>1750] = color['Color'].quantile(0.95)

#Turbiedad
datos_turbiedad.shape
datos_turbiedad.info()
datos_turbiedad.isnull().sum()
datos_turbiedad['Valor'].describe()

plt.boxplot(datos_turbiedad['Valor'])
datos_turbiedad['Valor'][datos_turbiedad['Valor']<0] = 0
datos_turbiedad['Valor'][datos_turbiedad['Valor']==0] = datos_turbiedad['Valor'].median()
datos_turbiedad['Valor'].quantile(0.95)
datos_turbiedad['Valor'][datos_turbiedad['Valor']>350] = datos_turbiedad['Valor'].quantile(0.95)
plt.boxplot(datos_turbiedad['Valor'])

#Conductividad
plt.boxplot(datos_conductividad['Valor'])
datos_conductividad.shape
datos_conductividad.info()
datos_conductividad.isnull().sum()
datos_conductividad['Valor'].describe()

datos_conductividad['Valor'][datos_conductividad['Valor']<0] = 0
datos_conductividad['Valor'][datos_conductividad['Valor']==0] = datos_conductividad['Valor'].mean()
plt.boxplot(datos_conductividad['Valor'])

datos_conductividad['Valor'].quantile(0.05)
datos_conductividad['Valor'][datos_conductividad['Valor']<62] = datos_conductividad['Valor'].quantile(0.05)

#PH
datos_ph.shape
datos_ph.info()
datos_ph.isnull().sum()
datos_ph['Valor'].describe()

datos_ph['Valor'][datos_ph['Valor']<0] = 0
datos_ph['Valor'][datos_ph['Valor']==0] = datos_ph['Valor'].mean()
plt.boxplot(datos_ph['Valor'])

datos_ph['Valor'].quantile(0.05)
datos_ph['Valor'][datos_ph['Valor']<6.8] = datos_ph['Valor'].quantile(0.05)
datos_ph['Valor'].quantile(0.98)
datos_ph['Valor'][datos_ph['Valor']>8.5] = datos_ph['Valor'].quantile(0.98)

#Temperatura
datos_temperatura.shape
datos_temperatura.info()
datos_temperatura.isnull().sum()
datos_temperatura['Temperatura'].describe()

datos_temperatura['Temperatura'][datos_temperatura['Temperatura']<0] = datos_temperatura['Temperatura'].mean()

#Precipitación
datos_precipitacion.shape
datos_precipitacion.info()
datos_precipitacion.isnull().sum()
datos_precipitacion['P1'].describe()

 datos_precipitacion['P1'].quantile(0.95)


#Dosis
dosis.shape
dosis.info()
dosis.isnull().sum()
dosis['Dosis'].describe()

dosis['Dosis'].quantile(0.02)
dosis['Dosis'][dosis['Dosis']<8] = dosis['Dosis'].quantile(0.02)

dosis['Dosis'].quantile(0.95)
dosis['Dosis'][dosis['Dosis']>30] = dosis['Dosis'].quantile(0.95)

#Caudal
caudal.shape
caudal.info()
caudal.isnull().sum()
caudal['Caudal'].describe()

caudal['Caudal'].quantile(0.02)
caudal['Caudal'][caudal['Caudal']<0] = caudal['Caudal'].quantile(0.02)

caudal['Caudal'].quantile(0.95)

plt.boxplot(caudal['Caudal'])
plt.hist(caudal['Caudal'])

##Analítica descriptiva
##Turbiedad
datos_turbiedad['Fecha'] = pd.to_datetime(datos_turbiedad['Fecha'],format='%Y%m%d%hh%mm%ss')
datos_turbiedad['Calendario'] = pd.DatetimeIndex(datos_turbiedad['Fecha']).year
datos_turbiedad['Mes'] = pd.DatetimeIndex(datos_turbiedad['Fecha']).month
datos_turbiedad['Dia'] = pd.DatetimeIndex(datos_turbiedad['Fecha']).weekday+1
datos_turbiedad['Hora'] = pd.DatetimeIndex(datos_turbiedad['Fecha']).hour
datos_turbiedad["Periodo"] = datos_turbiedad["Calendario"].astype(str) + '-' + datos_turbiedad["Mes"].astype(str)
datos_turbiedad['Tiempo'] = datos_turbiedad["Calendario"].astype(str) + '-' + datos_turbiedad["Mes"].astype(str) + '-' + datos_turbiedad["Dia"].astype(str) + '-' + datos_turbiedad["Hora"].astype(str)
datos_turbiedad.info()

turbiedad_df = datos_turbiedad.groupby(['Tiempo'],as_index=False)['Valor'].mean()
turbiedad_df['Tiempo'] = pd.to_datetime(turbiedad_df['Tiempo'],format="%Y-%m-%d-%H")


#Análisis turbiedad por mes,dia,hora
datos_turbiedad.groupby(['Mes'])['Valor'].mean().plot(kind='bar')
plt.title('Análisis Turbiedad por Mes')
plt.ylabel('Turbiedad')


datos_turbiedad.groupby(['Dia'])['Valor'].mean().plot(kind='bar')


datos_turbiedad.groupby(['Hora'])['Valor'].mean().plot(kind='bar')
plt.title('Análisis Turbiedad por Hora')
plt.ylabel('Turbiedad')


datos_turbiedad.groupby(['Periodo'])['Valor'].mean().plot()
plt.title('Tendencia turbiedad 2016-2021')
plt.ylabel('Turbiedad')



turb_cal=datos_turbiedad[['Calendario','Valor']]
turb_cal.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


turb_hr=datos_turbiedad[['Hora','Valor']]
turb_hr.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


turb_mes=datos_turbiedad[['Mes','Valor']]
turb_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


##Dosis
dosis['Fecha'] = pd.to_datetime(dosis['Fecha'],format='%Y/%m/%d %H:%M:%S')
dosis['Calendario'] = pd.DatetimeIndex(dosis['Fecha']).year
dosis['Mes'] = pd.DatetimeIndex(dosis['Fecha']).month
dosis['Dia'] = pd.DatetimeIndex(dosis['Fecha']).weekday+1
dosis['Hora'] = pd.DatetimeIndex(dosis['Fecha']).hour
dosis["Periodo"] = dosis["Calendario"].astype(str) + '-' + dosis["Mes"].astype(str)
dosis['Tiempo'] = dosis["Calendario"].astype(str) + '-' + dosis["Mes"].astype(str) + '-' + dosis["Dia"].astype(str) + '-' + dosis["Hora"].astype(str)
dosis.info()

dosis_df = dosis.groupby(['Tiempo'],as_index=False)['Dosis'].mean()
dosis_df['Tiempo'] = pd.to_datetime(dosis_df['Tiempo'],format="%Y-%m-%d-%H")

#Análisis dosis por mes,dia,hora

#Tendencia
dosis.groupby(['Periodo'])['Dosis'].mean().plot()
plt.title('Tendencia dosis 2016-2021',Fontsize=20)
plt.ylabel('Dosis',Fontsize=15)
plt.xlabel('Periodo',Fontsize=15)

dosis_cal=dosis[['Calendario','Dosis']]
dosis_cal.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)
plt.title('Tendencia dosis 2016-2021',Fontsize=20)
plt.ylabel('Dosis',Fontsize=15)
plt.xlabel('Periodo',Fontsize=15)

##Mes
dosis.groupby(['Mes'])['Dosis'].mean().plot(kind='bar')
plt.title('Análisis dosis por Mes',Fontsize=20)
plt.ylabel('Dosis',Fontsize=15)
plt.xlabel('Mes',Fontsize=15)

dosis_mes=dosis[['Mes','Dosis']]
dosis_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)
plt.title('Análisis dosis por Mes',Fontsize=20)
plt.ylabel('Dosis',Fontsize=15)
plt.xlabel('Mes',Fontsize=15)


dosis.groupby(['Dia'])['Dosis'].mean().plot(kind='bar')

dosis.groupby(['Hora'])['Dosis'].mean().plot(kind='bar')
plt.title('Análisis dosis por Hora',Fontsize=20)
plt.ylabel('Dosis',Fontsize=15)
plt.xlabel('Hora',Fontsize=15)

dosis_hr=dosis[['Hora','Dosis']]
dosis_hr.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)
plt.title('Análisis dosis por Hora',Fontsize=20)
plt.ylabel('Dosis',Fontsize=15)
plt.xlabel('Hora',Fontsize=15)







##Conductividad
datos_conductividad['Fecha'] = pd.to_datetime(datos_conductividad['Fecha'],format='%Y%m%d%hh%mm%ss')
datos_conductividad['Calendario'] = pd.DatetimeIndex(datos_conductividad['Fecha']).year
datos_conductividad['Mes'] = pd.DatetimeIndex(datos_conductividad['Fecha']).month
datos_conductividad['Dia'] = pd.DatetimeIndex(datos_conductividad['Fecha']).weekday+1
datos_conductividad['Hora'] = pd.DatetimeIndex(datos_conductividad['Fecha']).hour
datos_conductividad["Periodo"] = datos_conductividad["Calendario"].astype(str) + '-' + datos_conductividad["Mes"].astype(str)
datos_conductividad['Tiempo'] = datos_conductividad["Calendario"].astype(str) + '-' + datos_conductividad["Mes"].astype(str) + '-' + datos_conductividad["Dia"].astype(str) + '-' + datos_conductividad["Hora"].astype(str)
datos_conductividad.head()

conductividad_df = datos_conductividad.groupby(['Tiempo'],as_index=False)['Valor'].mean()
conductividad_df['Tiempo'] = pd.to_datetime(conductividad_df['Tiempo'],format="%Y-%m-%d-%H")

#Análisis conductividad por mes,dia,hora
datos_conductividad.groupby(['Mes'])['Valor'].mean().plot(kind='bar')
plt.title('Análisis conductividad por Mes')
plt.ylabel('Conductividad')


datos_conductividad.groupby(['Periodo'])['Valor'].mean().plot()
plt.title('Tendencia conductividad 2016-2021')
plt.ylabel('Conductividad')

cond_cal=datos_conductividad[['Calendario','Valor']]
cond_cal.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)



datos_conductividad.groupby(['Dia'])['Valor'].mean().plot(kind='bar')



datos_conductividad.groupby(['Hora'])['Valor'].mean().plot(kind='bar')
plt.title('Análisis conductividad por Hora')
plt.ylabel('Conductividad')

cond_mes=datos_conductividad[['Mes','Valor']]
cond_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)

cond_hora=datos_conductividad[['Hora','Valor']]
cond_hora.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)



##Caudal
caudal['Fecha'] = pd.to_datetime(caudal['Fecha'],format='%Y/%m/%d %H:%M:%S')
caudal['Calendario'] = pd.DatetimeIndex(caudal['Fecha']).year
caudal['Mes'] = pd.DatetimeIndex(caudal['Fecha']).month
caudal['Dia'] = pd.DatetimeIndex(caudal['Fecha']).weekday+1
caudal['Hora'] = pd.DatetimeIndex(caudal['Fecha']).hour
caudal["Periodo"] = caudal["Calendario"].astype(str) + '-' + caudal["Mes"].astype(str)
caudal['Tiempo'] = caudal["Calendario"].astype(str) + '-' + caudal["Mes"].astype(str) + '-' + caudal["Dia"].astype(str) + '-' + caudal["Hora"].astype(str)
caudal.head()

caudal_df = caudal.groupby(['Tiempo'],as_index=False)['Caudal'].mean()
caudal_df['Tiempo'] = pd.to_datetime(caudal_df['Tiempo'],format="%Y-%m-%d-%H")

#Análisis caudal por mes,dia,hora
caudal.groupby(['Mes'])['Caudal'].mean().plot(kind='bar')
plt.title('Análisis conductividad por Mes')
plt.ylabel('Conductividad')

caudal_mes=caudal[['Mes','Caudal']]
caudal_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)

caudal.groupby(['Periodo'])['Caudal'].mean().plot()
plt.title('Tendencia caudal 2016-2021')
plt.ylabel('Caudal')

caudal_cal=caudal[['Calendario','Caudal']]
caudal_cal.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


caudal.groupby(['Dia'])['Caudal'].mean().plot(kind='bar')
plt.title('Caudal por día de la semana')
plt.ylabel('Caudal')


caudal.groupby(['Hora'])['Caudal'].mean().plot(kind='bar')
plt.title('Análisis caudal por Hora',Fontsize=20)
plt.ylabel('Caudal',Fontsize=15)
plt.xlabel('Hora',Fontsize=15)


caudal_hora=caudal[['Hora','Caudal']]
caudal_hora.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


##PH
datos_ph['Fecha'] = pd.to_datetime(datos_ph['Fecha'],format='%Y%m%d%hh%mm%ss')
datos_ph['Calendario'] = pd.DatetimeIndex(datos_ph['Fecha']).year
datos_ph['Mes'] = pd.DatetimeIndex(datos_ph['Fecha']).month
datos_ph['Dia'] = pd.DatetimeIndex(datos_ph['Fecha']).weekday+1
datos_ph['Hora'] = pd.DatetimeIndex(datos_ph['Fecha']).hour
datos_ph["Periodo"] = datos_ph["Calendario"].astype(str) + '-' + datos_ph["Mes"].astype(str)
datos_ph['Tiempo'] = datos_ph["Calendario"].astype(str) + '-' + datos_ph["Mes"].astype(str) + '-' + datos_ph["Dia"].astype(str) + '-' + datos_ph["Hora"].astype(str)

ph_df = datos_ph.groupby(['Tiempo'],as_index=False)['Valor'].mean()
ph_df['Tiempo'] = pd.to_datetime(ph_df['Tiempo'],format="%Y-%m-%d-%H")

#Análisis ph por mes,dia,hora
datos_ph.groupby(['Mes'])['Valor'].mean().plot(kind='bar')
plt.title('Análisis PH por mes')
plt.ylabel('PH')


datos_ph.groupby(['Periodo'])['Valor'].mean().plot()
plt.title('Tendencia PH 2016-2021')
plt.ylabel('PH')

ph_cal=datos_ph[['Calendario','Valor']]
ph_cal.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


datos_ph.groupby(['Dia'])['Valor'].mean().plot(kind='bar')


datos_ph.groupby(['Hora'])['Valor'].mean().plot(kind='bar')
plt.title('Análisis PH por hora')
plt.ylabel('PH')


ph_mes=datos_ph[['Mes','Valor']]
ph_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


ph_hora=datos_ph[['Hora','Valor']]
ph_hora.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=False, showbox=True,            
                 showfliers=False)



##Temperatura
##Turbiedad
datos_temperatura['Fecha'] = pd.to_datetime(datos_temperatura['fecha_hora'],format='%Y%m%d%hh%mm%ss')
datos_temperatura['Calendario'] = pd.DatetimeIndex(datos_temperatura['Fecha']).year
datos_temperatura['Mes'] = pd.DatetimeIndex(datos_temperatura['Fecha']).month
datos_temperatura['Dia'] = pd.DatetimeIndex(datos_temperatura['Fecha']).weekday+1
datos_temperatura['Hora'] = pd.DatetimeIndex(datos_temperatura['Fecha']).hour
datos_temperatura["Periodo"] = datos_temperatura["Calendario"].astype(str) + '-' + datos_temperatura["Mes"].astype(str)
datos_temperatura['Tiempo'] = datos_temperatura["Calendario"].astype(str) + '-' + datos_temperatura["Mes"].astype(str) + '-' + datos_temperatura["Dia"].astype(str) + '-' + datos_temperatura["Hora"].astype(str)
datos_temperatura.info()

temperatura_df = datos_temperatura.groupby(['Tiempo'],as_index=False)['Temperatura'].mean()
temperatura_df['Tiempo'] = pd.to_datetime(temperatura_df['Tiempo'],format="%Y-%m-%d-%H")

#Análisis temperatura por mes,dia,hora
datos_temperatura.groupby(['Mes'])['Temperatura'].mean().plot(kind='bar')
plt.title('Análisis Temperatura por Mes')
plt.ylabel('Temperatura')

datos_temperatura.groupby(['Dia'])['Temperatura'].mean().plot(kind='bar')

datos_temperatura.groupby(['Hora'])['Temperatura'].mean().plot(kind='bar')
plt.title('Análisis Temperatura por Hora')
plt.ylabel('Temperatura')

datos_temperatura.groupby(['Periodo'])['Temperatura'].mean().plot()
plt.title('Tendencia temperatura 2016-2021')
plt.ylabel('Temperatura')

temp_cal=datos_temperatura[['Calendario','Temperatura']]
temp_cal.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)

temp_mes=datos_temperatura[['Mes','Temperatura']]
temp_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=False, showbox=True,            
                 showfliers=False)


temp_hora=datos_temperatura[['Hora','Temperatura']]
temp_hora.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)



#Análisis precipitacion por mes,dia,hora
datos_precipitacion['Fecha'] = pd.to_datetime(datos_precipitacion['fecha_hora'],format='%Y%m%d%hh%mm%ss')
datos_precipitacion['Calendario'] = pd.DatetimeIndex(datos_precipitacion['Fecha']).year
datos_precipitacion['Mes'] = pd.DatetimeIndex(datos_precipitacion['Fecha']).month
datos_precipitacion['Dia'] = pd.DatetimeIndex(datos_precipitacion['Fecha']).weekday+1
datos_precipitacion['Hora'] = pd.DatetimeIndex(datos_precipitacion['Fecha']).hour
datos_precipitacion["Periodo"] = datos_precipitacion["Calendario"].astype(str) + '-' + datos_precipitacion["Mes"].astype(str)
datos_precipitacion['Tiempo'] = datos_precipitacion["Calendario"].astype(str) + '-' + datos_precipitacion["Mes"].astype(str) + '-' + datos_precipitacion["Dia"].astype(str) + '-' + datos_precipitacion["Hora"].astype(str)
datos_precipitacion.info()
datos_precipitacion.head()

precipitacion_df = datos_precipitacion.groupby(['Tiempo'],as_index=False)['P1'].mean()
precipitacion_df['Tiempo'] = pd.to_datetime(precipitacion_df['Tiempo'],format="%Y-%m-%d-%H")

datos_precipitacion.groupby(['Calendario'])['P1'].mean().plot(kind='bar')
plt.title('Análisis Precipitación por Año')
plt.ylabel('Precipitación')

datos_precipitacion.groupby(['Mes'])['P1'].mean().plot(kind='bar')
plt.title('Análisis Precipitación por Mes')
plt.ylabel('Precipitación')

datos_precipitacion.groupby(['Dia'])['P1'].mean().plot(kind='bar')

datos_precipitacion.groupby(['Hora'])['P1'].mean().plot(kind='bar')
plt.title('Análisis Precipitación por Hora')
plt.ylabel('Precipitación')

datos_precipitacion.groupby(['Periodo'])['P1'].mean().plot()
plt.title('Tendencia precipitación 2016-2021')
plt.ylabel('Precipitación')

prec_mes=datos_precipitacion[['Mes','P1']]
prec_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


prec_hora=datos_precipitacion[['Hora','P1']]
prec_hora.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)



#Análisis color por mes,dia,hora
color['Fecha'] = pd.to_datetime(color['Fecha Hora'],format='%Y%m%d%hh%mm%ss')
color['Calendario'] = pd.DatetimeIndex(color['Fecha']).year
color['Mes'] = pd.DatetimeIndex(color['Fecha']).month
color['Dia'] = pd.DatetimeIndex(color['Fecha']).weekday+1
color['Hora'] = pd.DatetimeIndex(color['Fecha']).hour
color["Periodo"] = color["Calendario"].astype(str) + '-' + color["Mes"].astype(str)
color['Tiempo'] = color["Calendario"].astype(str) + '-' + color["Mes"].astype(str) + '-' + color["Dia"].astype(str) + '-' + color["Hora"].astype(str)
color.info()
color.head()

color_df = color.groupby(['Tiempo'],as_index=False)['Color'].mean()
color_df['Tiempo'] = pd.to_datetime(color_df['Tiempo'],format="%Y-%m-%d-%H")

color.groupby(['Calendario'])['Color'].mean().plot(kind='bar')
plt.title('Análisis Color por Año')
plt.ylabel('Color')

color.groupby(['Mes'])['Color'].mean().plot(kind='bar')
plt.title('Análisis Color por Mes')
plt.ylabel('Color')

color.groupby(['Dia'])['Color'].mean().plot(kind='bar')

color.groupby(['Hora'])['Color'].mean().plot(kind='bar')
plt.title('Análisis Color por Hora')
plt.ylabel('Color')

color.groupby(['Periodo'])['Color'].mean().plot()
plt.title('Tendencia Color 2016-2021')
plt.ylabel('Color')

color_mes=color[['Mes','Color']]
color_mes.boxplot(by = 'Mes', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


color_hora=color[['Hora','Color']]
color_hora.boxplot(by = 'Hora', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)


color_hora=color[['Calendario','Color']]
color_hora.boxplot(by = 'Calendario', meanline=True, showmeans=True, showcaps=True, showbox=True,            
                 showfliers=False)




#Análisis multivariado
##Nuevos DF con solo las horas
turbiedad_df.rename(columns={'Valor':'Turbiedad'},inplace=True)
conductividad_df.rename(columns={'Valor':'Conductividad'},inplace=True)
ph_df.rename(columns={'Valor':'Ph'},inplace=True)
precipitacion_df.rename(columns={'P1':'Precipitacion'},inplace=True)

valor_turbiedad = turbiedad_df.loc[:,['Tiempo','Turbiedad']]
valor_color = color_df.loc[:,['Tiempo','Color']]
valor_conductividad = conductividad_df.loc[:,['Tiempo','Conductividad']]
valor_ph = ph_df.loc[:,['Tiempo','Ph']]
valor_temperatura = temperatura_df.loc[:,['Tiempo','Temperatura']]
valor_precipitacion = precipitacion_df.loc[:,['Tiempo','Precipitacion']]
valor_dosis = dosis_df.loc[:,['Tiempo','Dosis']]
valor_caudal = caudal_df.loc[:,['Tiempo','Caudal']]

dataframe = pd.concat([valor_turbiedad,valor_conductividad,valor_ph,valor_temperatura,valor_precipitacion,valor_color,valor_caudal,valor_dosis],axis=1)
dataframe.shape
dataframe.columns.tolist()
dataframe = dataframe.iloc[:,[0,1,3,5,7,9,11,13,15]]
dataframe.dropna(axis=0, how='any',inplace=True)


dataframe.to_excel('Dataframe_potabilizacion.xlsx')

#Estandarizacion
escaler = StandardScaler()
df_esc = escaler.fit_transform(dataframe.iloc[:,1:])
df_esc = pd.DataFrame(df_esc)
df_esc.to_excel('dataframe_potabilizacion_est.xlsx')
##Analísis de correlacion

correlacion_df = dataframe.corr()
sns.heatmap(correlacion_df, annot=True)
plt.show()

plt.scatter(dataframe['Color'],dataframe['Temperatura'])
plt.title('Gráfica de dispersión Color vs Temperatura',fontsize=20)
plt.xlabel('Color',fontsize=20)
plt.ylabel('Temperatura',fontsize=20)

plt.scatter(dataframe['Turbiedad'],dataframe['Color'])
plt.title('Gráfica de dispersión Turbiedad vs Color',fontsize=20)
plt.xlabel('Turbiedad',fontsize=20)
plt.ylabel('Color',fontsize=20)

plt.scatter(dataframe['Ph'],dataframe['Temperatura'])
plt.title('Gráfica de dispersión Ph vs Temperatura',fontsize=20)
plt.xlabel('Ph',fontsize=20)
plt.ylabel('Temperatura',fontsize=20)

plt.scatter(dataframe['Turbiedad'],dataframe['Precipitacion'])
plt.title('Gráfica de dispersión Turbiedad vs Precipitacion',fontsize=20)
plt.xlabel('Turbiedad',fontsize=20)
plt.ylabel('Precipitacion',fontsize=20)


#Gráfica
sns.set_theme(style="ticks")
sns.pairplot(dataframe, hue="Turbiedad")

##Scatter 3D
X=dataframe['Turbiedad']
Y=dataframe['Precipitacion']
Z=dataframe['Temperatura']

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter3D(X,Y,c=X,cmap='Set1')
plt.show




##Análisis variables por pares
#Temperatura y turbiedad
turb_temperatura = pd.merge(left=turbiedad_df, right=Temperatura, left_on='Tiempo', right_on='Tiempo')
turb_ph.rename(columns={'Valor_x':'Turbiedad','Valor_y':'PH'},inplace=True)

#Turbiedad y PH
turb_ph = pd.merge(left=datos_turbiedad, right=valor_ph, left_on='Fecha', right_on='Fecha')
turb_ph.rename(columns={'Valor_x':'Turbiedad','Valor_y':'PH'},inplace=True)

#Turbiedad y Conductividad
turb_cond = pd.merge(left=datos_turbiedad, right=valor_conductividad, left_on='Fecha', right_on='Fecha')
turb_cond.rename(columns={'Valor_x':'Turbiedad','Valor_y':'Conductividad'},inplace=True)

##Grafica turbiedad vs conductividad 1
sns.set_theme(style="ticks")
g = sns.JointGrid(data=turb_cond, x="Turbiedad", y="Conductividad", marginal_ticks=True)
g.ax_joint.set(yscale="log",xscale="Log")
cax = g.fig.add_axes([.15, .55, .02, .2])
g.plot_joint(
    sns.histplot, discrete=(True, False),
    cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax
)
g.plot_marginals(sns.histplot, element="step", color="#03012d")


##Grafica turbiedad vs conductividad 2
sns.set_theme(style="ticks")
sns.jointplot(x='Turbiedad', y='Conductividad',data=turb_cond, kind="hex", color="#4CB391")

sns.pairplot(turb_cond.drop(['Conductividad','Calendario'],axis=1),hue='Turbiedad',aspect=3);

sns.pairplot(turb_cond.drop(['Turbiedad','Calendario'],axis=1),hue='Conductividad',aspect=3);

sns.pairplot(turb_cond.drop(['Turbiedad','Calendario'],axis=1),hue='Conductividad',aspect=3);

"""Análisis predictivo"""
###
###
###
###
###
###



datos = pd.read_excel("Dataframe_potabilizacion.xlsx")

#datos.drop(['Dosis'],axis=1, inplace=True)
datos.drop(['Tiempo'],axis=1, inplace=True)

datos.dropna(axis=0,how='any',inplace=True)

datos.shape
datos.columns.tolist()

##Turbiedad
datos['Turbiedad'].describe()
plt.boxplot(datos['Turbiedad'])
plt.hist(datos['Turbiedad'])

##Conductividad
datos['Conductividad'].describe()
plt.boxplot(datos['Conductividad'])
plt.hist(datos['Conductividad'])

##Caudal
datos['Caudal'].describe()
plt.boxplot(datos['Caudal'])
plt.hist(datos['Caudal'])


##PH
datos['Ph'].describe()
plt.boxplot(datos['Ph'])
plt.hist(datos['Ph'])


##Dosis 
datos['Dosis'].describe()
plt.boxplot(datos['Dosis'])
plt.hist(datos['Dosis'])


##Color
datos['Color'].describe()
plt.boxplot(datos['Color'])
plt.hist(datos['Color'])


###Matriz de correlación
correlacion = datos.corr()
sns.heatmap(correlacion, annot=True)
plt.show()


datos.columns.tolist()
datos.shape

#X Y Y
X = datos.iloc[:,0:11]
Y = datos['Dosis']


#Estandarizacion
escaler = StandardScaler()
X_esc = escaler.fit_transform(X)

#Feature selection
sel = SelectKBest(score_func=mutual_info_regression, k='all')
sel.fit(X_esc, Y)
sel

for i in range(len(sel.scores_)):
	print('Feature %d: %f' % (i, sel.scores_[i]))
plt.bar([i for i in range(len(sel.scores_))], sel.scores_)
plt.show()

plt.figure(figsize=(26,24),facecolor='w', edgecolor='k',num=1)
feat_importances = pd.Series(sel.scores_, index= X.columns)
feat_importances.nlargest(20).plot(kind='barh')
parameters = {'axes.labelsize': 35,
          'axes.titlesize': 35}
plt.rcParams.update(parameters)
plt.title('Importancia de las variables')
plt.xlabel('% Importancia', fontsize=30)
plt.ylabel('Variables', fontsize=30)


##Boruta
model = xgb.XGBRegressor()

feat_selector = BorutaPy(model, n_estimators='auto',verbose=2, random_state=1)

feat_selector.fit(X_esc,Y)

print(feat_selector.support_)
print(feat_selector.ranking_)

feat_selector

X_filtered = feat_selector.transform(X_esc)

feature_names = np.array(X.columns)

feature_ranks = list(zip(feature_names,
                         feat_selector.ranking_,
                         feat_selector.support_))
feature_ranks


#Quito características que no agregan valor
datos.drop(['Dia'],axis=1, inplace=True)
datos.drop(['Temperatura'],axis=1, inplace=True)
datos.drop(['Precipitacion'],axis=1, inplace=True)

##Guardar en excel
datos.to_excel('Datos_coagulante_dosis.xlsx')

##Creacion variable dummy
datos['FRANJA'] = datos['FRANJA'].astype('category')
datos.info()

datos = pd.get_dummies(datos,prefix='Franja')

datos.columns.tolist()
datos.shape

#Nuevas X Y Y
X_columns = ['Año','Mes','Hora',
             'Turbiedad','Conductividad',
             'Ph','Color','Caudal']

datos.dropna(axis=0,how='any',inplace=True)

X = datos.loc[:,X_columns]
Y = datos['Dosis']

##Estandarización de variabiónes
escaler = StandardScaler()
X_esc = escaler.fit_transform(X)

##Guardar dataframe estandarizado
escaler = StandardScaler()
df_esc = escaler.fit_transform(datos.iloc[:,0:8])
df_esc = pd.DataFrame(df_esc)
df_esc.to_excel('Datos_coagulante_dosis_est.xlsx')


#Guardar modelo de estandarizacion
esc = escaler.fit(X)
joblib.dump(esc,'estandarizacion.pkl')

#Division de datos
X_train, X_test, y_train, y_test = train_test_split(X_esc,Y,test_size=0.2)

##Entrenamiento modelos
#Regresión lineal
lrm = LinearRegression()

lrm.fit(X_train,y_train)

y_pred_lr = lrm.predict(X_test)
y_pred_train_lr = lrm.predict(X_train)

print(lrm.coef_)

print(lrm.intercept_)

math.sqrt(mean_squared_error(y_train, y_pred_train_lr))
math.sqrt(mean_squared_error(y_test, y_pred_lr))

mean_absolute_error(y_train, y_pred_train_lr)
mean_absolute_error(y_test, y_pred_lr)

r2_score(y_train,y_pred_train_lr)
r2_score(y_test,y_pred_lr)


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_lr, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de RL',Fontsize=20)
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train_lr, edgecolors=(0, 0, 0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de RL',Fontsize=20)
plt.show()

##Arbol de decisión
modelTree = DecisionTreeRegressor()

min_samples_leaf = [2,10,50,100]
max_depth = [None,10,20,50]

param_grid = dict(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
grid = GridSearchCV(estimator=modelTree, param_grid=param_grid, n_jobs=-1, cv=3)
grid.fit(X_train,y_train)

modelTree = grid.best_estimator_

print(grid.best_params_)

modelTree.fit(X_train,y_train)

y_pred_dt = modelTree.predict(X_test)
y_pred_train_dt = modelTree.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_dt))
math.sqrt(mean_squared_error(y_test, y_pred_dt))

mean_absolute_error(y_train, y_pred_train_dt)
mean_absolute_error(y_test, y_pred_dt)

r2_score(y_test,y_pred_dt)
r2_score(y_train,y_pred_train_dt)

fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train_dt, edgecolors=(0, 0, 0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de Árboles',Fontsize=20)
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_dt, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de Árboles',Fontsize=20)
plt.show()

#Gráfica Árbol
datos.columns.tolist()
names_x = ['Año','Mes','Hora','Turbiedad','Conductividad',
         'Ph','Color','Caudal']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(modelTree,
               fontsize=6,
               feature_names = names_x,
               filled = True);

plt.figure(figsize=(12,12))
tree.plot_tree(modelTree, fontsize=10,feature_names = names_x,filled = True)
plt.show()


##Support vector Machine
C = [0.1,1,10]
kernel = ['linear','rbf']
gamma = ['scale','auto',0.01,5]

svmrm = SVR()

param_grid = dict(C=C, kernel=kernel, gamma=gamma)
grid = GridSearchCV(estimator=svmrm, param_grid=param_grid, n_jobs=-1, cv=3)
grid.fit(X_train,y_train)

svmrm = grid.best_estimator_

svmrm.fit(X_train,y_train)

y_pred_svm = svmrm.predict(X_test)
y_pred_train_svm = svmrm.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_svm))
math.sqrt(mean_squared_error(y_test, y_pred_svm))

mean_absolute_error(y_train, y_pred_train_svm)
mean_absolute_error(y_test, y_pred_svm)

r2_score(y_test,y_pred_svm)
r2_score(y_train,y_pred_train_svm)

fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train_svm, edgecolors=(0, 0, 0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de SVM',Fontsize=20)
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_svm, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de SVM',Fontsize=20)
plt.show()

##Bosque aleatorio
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.best_params_

rf_random.fit(params, X_train,y_train)

y_pred_rf = rf_random.predict(X_test)
y_pred_train_rf = rf_random.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_rf))
math.sqrt(mean_squared_error(y_test, y_pred_rf))

mean_absolute_error(y_train, y_pred_train_rf)
mean_absolute_error(y_test, y_pred_rf)

r2_score(y_train,y_pred_train_rf)
r2_score(y_test, y_pred_rf)


fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train_rf, edgecolors=(0, 0, 0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de RF',Fontsize=30)
plt.show()


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_rf, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de RF',Fontsize=15)
plt.show()


##Redes neuronales
mlp_gs = MLPRegressor(max_iter=1000)

parameter_space = {
    'hidden_layer_sizes': [(10,300,10),(20,)],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.01],
    'learning_rate': ['constant','adaptive'],
}

clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)

clf.fit(X_train, y_train)

clf.get_params()

y_pred_clf = clf.predict(X_test)
y_pred_train_clf = clf.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_clf))
math.sqrt(mean_squared_error(y_test, y_pred_rf))

mean_absolute_error(y_train, y_pred_train_clf)
mean_absolute_error(y_test, y_pred_clf)

r2_score(y_train,y_pred_train_clf)
r2_score(y_test, y_pred_clf)


##Bagging
bg = BaggingRegressor(DecisionTreeRegressor(),max_samples=0.5,max_features=1.0,n_estimators=10)
bg.fit(X_train,y_train)

y_pred_bg = bg.predict(X_test)
y_pred_train_bg = bg.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_bg))
math.sqrt(mean_squared_error(y_test, y_pred_bg))

mean_absolute_error(y_train, y_pred_train_bg)
mean_absolute_error(y_test, y_pred_bg)

r2_score(y_train,y_pred_train_bg)
r2_score(y_test, y_pred_bg)

##Guardar modelo
joblib.dump(bg,'modelo_entrenado_bg.pkl')
modelo = joblib.load('modelo_entrenado_bg.pkl')
modelo.score(X_test,y_test)


#Bostrap
bt = AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=5,learning_rate=0.1)
bt.fit(X_train,y_train)

y_pred_bt = bt.predict(X_test)
y_pred_train_bt = bt.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_bt))
math.sqrt(mean_squared_error(y_test, y_pred_bt))

mean_absolute_error(y_train, y_pred_train_bt)
mean_absolute_error(y_test, y_pred_bt)

r2_score(y_train,y_pred_train_bt)
r2_score(y_test, y_pred_bt)


##XBGBoostRegressor
xgb = xgb.XGBRegressor()

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_pred_train_xgb = xgb.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
math.sqrt(mean_squared_error(y_test, y_pred_xgb))

mean_absolute_error(y_train, y_pred_train_xgb)
mean_absolute_error(y_test, y_pred_xgb)

r2_score(y_train,y_pred_train_xgb)
r2_score(y_test, y_pred_xgb)

##LightBoosting
ltb = ltb.LGBMRegressor()

#Sin parametros
ltb.fit(X_train, y_train)

y_pred_ltb = ltb.predict(X_test)
y_pred_train_ltb = ltb.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_ltb))
math.sqrt(mean_squared_error(y_test, y_pred_ltb))

mean_absolute_error(y_train, y_pred_train_ltb)
mean_absolute_error(y_test, y_pred_ltb)

r2_score(y_train,y_pred_train_ltb)
r2_score(y_test, y_pred_ltb)

fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train_ltb, edgecolors=(0, 0, 0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis')
ax.set_ylabel('Predicted dosis')
ax.set_title('Dosis real vs Predicción modelo de ltb')
plt.show()


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_ltb, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis')
ax.set_ylabel('Predicted dosis')
ax.set_title('Dosis real vs Predicción modelo de ltb')
plt.show()


##LightBoosting
min = 99999999999999999999999 
count = 0 
iterations = 150
for i in range(iterations):
    print('iteration number', count)
    count += 1
    d_train = lgb.Dataset(X_train, label=y_train)
    params = {} 
    params['learning_rate'] = np.random.uniform(0, 1)
    params['boosting_type'] = np.random.choice(['gbdt', 'dart', 'goss'])
    params['objective'] = 'regression'
    params['metric'] = 'mae'
    params['sub_feature'] = np.random.uniform(0, 1)
    params['num_leaves'] = np.random.randint(20, 300)
    params['min_data'] = np.random.randint(10, 100)
    params['max_depth'] = np.random.randint(5, 200)
    iterations = np.random.randint(10, 10000)
    print(params, iterations)
clr = lgb.train(params, d_train, iterations)
y_pred=clr.predict(X_test) #Create predictions on test set
mae=mean_absolute_error(y_pred,y_test)
print('MAE:', mae)
if mae < min:
    min = mae
    pp = params 
print(params)
print("*" * 50)
print('Minimum is: ', min)
print('Used params', pp)

params = {'learning_rate': 0.8525391624011257, 'boosting_type': 'dart', 'objective': 'regression', 'metric': 'mae', 'sub_feature': 0.2573149191243077, 'num_leaves': 281, 'min_data': 60, 'max_depth': 108}

train_data = lgb.Dataset(X_train,label=y_train)
valid_data = lgb.Dataset(X_test,label=y_test)

lgbm = lgb.train(params, train_data, 2500, valid_sets=valid_data,early_stopping_rounds= 50, verbose_eval=10)

y_pred_lgbm = lgbm.predict(X_test)
y_pred_train_lgbm = lgbm.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_lgbm))
math.sqrt(mean_squared_error(y_test, y_pred_lgbm))

mean_absolute_error(y_train, y_pred_train_lgbm)
mean_absolute_error(y_test, y_pred_lgbm)

r2_score(y_train,y_pred_train_lgbm)
r2_score(y_test, y_pred_lgbm)

##Ensemble liviano
vrl = VotingRegressor([('clf1',bg),('clf2',bt),('clf3',svmrm)])
vrl.fit(X_train, y_train)
y_pred_vrl = vrl.predict(X_test)
y_pred_train_vrl = vrl.predict(X_train)
math.sqrt(mean_squared_error(y_train, y_pred_train_vrl))
math.sqrt(mean_squared_error(y_test, y_pred_vrl))

mean_absolute_error(y_train, y_pred_train_vrl)
mean_absolute_error(y_test, y_pred_vrl)

r2_score(y_train,y_pred_train_vrl)
r2_score(y_test, y_pred_vrl)

joblib.dump(vrl,'modelo_entrenado_vrl.pkl')

##Ensemble
vr = VotingRegressor([('clf1',ltb),('clf2',clf),('clf3',bt),('clf4',bg),('clf5',svmrm),('clf6',rf_random)])

vr.fit(X_train, y_train)

y_pred_vr = vr.predict(X_test)
y_pred_train_vr = vr.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred_train_vr))
math.sqrt(mean_squared_error(y_test, y_pred_vr))

mean_absolute_error(y_train, y_pred_train_vr)
mean_absolute_error(y_test, y_pred_vr)

r2_score(y_train,y_pred_train_vr)
r2_score(y_test, y_pred_vr)

fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train_vr, edgecolors=(0, 0, 0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de vr',Fontsize=20)
plt.show()


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_vr, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual dosis',Fontsize=15)
ax.set_ylabel('Predicted dosis',Fontsize=15)
ax.set_title('Dosis real vs Predicción modelo de vr',Fontsize=20)
plt.show()

#Importancia de las variables

rf = ensemble.RandomForestRegressor()

single_rf = ensemble.RandomForestRegressor(n_estimators = 200, max_depth = 15)
single_rf.fit(X_esc, Y)
y_pred = single_rf.predict(X_esc)

colors = [plt.cm.twilight_shifted(i/float(len(X.columns)-1)) for i in range(len(X.columns))]
importances = single_rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns    
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices], color=colors, align="center")
plt.xticks(range(X.shape[1]), indices) #feature_names, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()
# list feature importance
important_features = pd.Series(data=single_rf.feature_importances_,index=X.columns)
important_features.sort_values(ascending=False,inplace=True)
print(important_features.head(15))


##Guardar modelo
joblib.dump(vr,'modelo_entrenado.pkl')

modelo = joblib.load('modelo_entrenado.pkl')
modelo.score(X_test,y_test)



