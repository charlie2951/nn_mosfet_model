# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:19:02 2021
Version 1.1, Rev date: 03/11/2021
Comment: Id-Vg is tuned, If-Vd not coming 
Version 1.2, Rev date: 13/11/2021, Drain current normalization formula changed
@author: KIIT
"""
#%%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#%%
def smooth(y):
    box_pts=2
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
#%%
df = pd.read_csv('training_data_eigen.csv')
df = shuffle(df)
#df["vg"].replace({0:1e-3}, inplace=True)
#df["qg"].replace({0:1e-22}, inplace=True)
#Normalize data before feeding to neural network
vg=np.ravel(df["vg"])/10
tch=np.ravel(df["tch"])/1e-9/10
tox=np.ravel(df["tox"])/1e-9/10
mch=np.ravel(df["mch"])/0.1
ub=np.ravel(df["ub"])/10
print(df)
X1=np.array([tch,mch,tox,ub,vg])
df0 = pd.DataFrame(data=X1.T,columns=["tch", "mch","tox","ub","vg"])
#print(df0)
X=df0.iloc[:,0:5]
print(X)
y=df.iloc[:,5:8]
y=y/10
print(y)
#%%
# Split train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# We'll use Keras to create a Neural network
model = tf.keras.Sequential()
model.add(keras.layers.Dense(32, activation='tanh', input_shape=(5,)))
#model.add(keras.layers.Dense(8,activation='tanh'))
#model.add(keras.layers.Dense(8,activation='tanh'))
#model.add(keras.layers.Dense(16,activation='tanh'))
#model.add(keras.layers.Dense(16,activation='tanh'))
model.add(keras.layers.Dense(3, activation='tanh'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history_1 = model.fit(X_train, y_train, epochs=1000, 
                        validation_data=(X_test, y_test))
#Save the model for future use and data extraction
#model.save('trained_model_58816163.h5')
#%%
# Exclude the first few epochs so the graph is easier to read
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']
epochs = range(1, len(loss) + 1)
SKIP = 300
plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
#Load trained model for testing
#Load trained model
#model = tf.keras.models.load_model('trained_model_321683.h5') #do not change
# New data input for testing
df_test = pd.read_csv('test03.csv')
#Normalize data before feeding to neural network
e0=np.ravel(df_test["e0"])
e1=np.ravel(df_test["e1"])
e2=np.ravel(df_test["e2"])
vg1=np.ravel(df_test["vg"])/10
tch1=np.ravel(df_test["tch"])/1e-9/10
tox1=np.ravel(df_test["tox"])/1e-9/10
mch1=np.ravel(df_test["mch"])/0.1
ub1=np.ravel(df_test["ub"])/10
X2=np.array([tch1,mch1,tox1,ub1,vg1])
df0_test = pd.DataFrame(data=X2.T,columns=["tch1", "mch1","tox1","ub1","vg1"])
#print(df0)
xval=df0_test.iloc[:,0:5]
# Predict the new dataset
y_pred = model.predict(xval)
#plot the result
vg_test1=np.ravel(df_test["vg"])
vg_test=vg_test1.reshape(-1,1)
#print(vg)
plt.plot(vg_test, e0, 'ro',vg_test, e1, 'mo', vg_test, e2, 'go')
#plt.plot(vg_test, e1, 'mo', label='Actual')
#plt.plot(vg_test, e2, 'go', label='Actual')
plt.plot(vg_test,10*(y_pred[:,0]), 'k',vg_test,10*(y_pred[:,1]), 'k',vg_test,10*(y_pred[:,2]), 'k')
#plt.plot(vg_test,10*(y_pred[:,0]), 'b', label='Predicted')
#plt.plot(vg_test,10*(y_pred[:,0]), 'b', label='Predicted')
#plt.yscale("log")
plt.title('Actual and Predicted Value')
plt.xlabel('Gate Voltage')
plt.ylabel('Sub-band Energies')
#plt.legend()
plt.show()
#Calculation of Cgg
q = 1.6e-19
mch = 0.05*9.1e-31
hbar = 6.626e-34/(2*3.14);
D=q*mch/(3.14*hbar**2) #2D density of states of channel material
phi_th=0.0259; #thermal voltage
q=1.6e-19; #Electronic charge
Ef=0 #Fermi energy fixed at 0eV
#Load predicted subband energy
E0p=10*np.ravel(y_pred[:,0])
E1p=10*np.ravel(y_pred[:,1])
E2p=10*np.ravel(y_pred[:,2])
#Calculation of Qinv using Fermi-Dirac function
Qinv_pred=q*D*phi_th*(np.log((1+np.exp((Ef-E0p)/phi_th)))+np.log((1+np.exp((Ef-E1p)/phi_th)))+np.log((1+np.exp((Ef-E2p)/phi_th))))
Qinv_actual=q*D*phi_th*(np.log((1+np.exp((Ef-e0)/phi_th)))+np.log((1+np.exp((Ef-e1)/phi_th)))+np.log((1+np.exp((Ef-e2)/phi_th))))
Cgg_actual=np.diff(Qinv_actual)/np.diff(vg_test1)
Cgg_pred=np.diff(Qinv_pred)/np.diff(vg_test1)
plt.plot(vg_test[1:len(vg_test1)], Cgg_actual, 'r', label='Actual')
plt.plot(vg_test[1:len(vg_test1)], Cgg_pred, 'bo', label='Pred')
#plt.plot(vg_test,10**(-1/y_pred), 'b', label='Predicted')
plt.legend()
plt.show()
print(Cgg_pred)
#plt.plot(vg_test, Qinv_actual, 'r', label='Actual')
#plt.plot(vg_test, Qinv_pred, 'bo', label='Pred')
#plt.plot(vg_test,10**(-1/y_pred), 'b', label='Predicted')
#plt.show()
Cgg_preddd=Cgg_pred
Cgg_pred1=Cgg_pred
#print(Cgg_pred1)

for i in range(0,len(Cgg_pred)-1):
    if Cgg_pred[i+1] < Cgg_pred[i]:
        Cgg_pred1[i+1]=Cgg_pred[i]
    else :
        Cgg_pred1[i+1]=Cgg_pred[i+1]
    Cgg_pred = Cgg_pred1
#print(Cgg_pred1)
plt.plot(vg_test[1:len(vg_test1)], Cgg_actual, 'r', label='Actual')
plt.plot(vg_test[1:len(vg_test1)], smooth(Cgg_pred1), 'b', label='Pred')
plt.show()
#%%
#calcularion of R^2 value
actual=Cgg_actual
predicted=Cgg_pred1
corr_matrix = np.corrcoef(actual, predicted)
corr = corr_matrix[0,1]
R_sq = corr**2
 
print(R_sq)
#%%Calculation of MSE
MSE = (1/len(Cgg_actual))*np.sum((Cgg_actual - Cgg_pred1)**2)
print(MSE)
print(rms_err/1e-6)
#Calculation of MAE

#calculation of RMSE error
rms_err = np.sqrt((1/len(Cgg_actual))*np.sum((Cgg_actual - Cgg_pred1)**2))
print(rms_err/1e-6)