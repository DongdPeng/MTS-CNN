from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from keras.models import Model
from keras.callbacks import EarlyStopping
from deepNF import build_MDA
import keras
import os.path as Path
import scipy.io as sio
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

def build_model(X, input_dims, arch, nf=0.5, std=1.0, mtype='mda', epochs=80, batch_size=64):
    if mtype == 'mda':
        model = build_MDA(input_dims, arch)
    else:
        print ("### Wrong model.")
    # corrupting the input
    noise_factor = nf
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.2)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = model.fit(X_train_noisy, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test_noisy, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)])
    mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

    return mid_model, history


# ### Main code starts here
if __name__ == "__main__":
    # Training settings

    model_type = 'mda'
    select_nets=['drug_ATC','drug_path','drug_side','drug_target','drug_enzyme','drug_smiles','drug_tran']
    select_net = ['drug_ATC','drug_path','drug_side','drug_target','drug_enzyme','drug_smiles','drug_tran']
    select_arch=[3]
    epochs = 80
    batch_size = 64
    nf = 0.5
    K = 3


    arch = {}
    arch['mda'] = {}
    arch['mda'] = {1: [1024],
                   2: [7 * 200, 600, 7 * 200],
                   3: [7 * 269, 7 * 100, 600, 7 * 100, 7 * 269],
                   4: [7 * 200, 7 * 100, 7 * 64, 200, 7 * 64, 7 * 100, 6 * 200]
                   }

    # load PPMI matrices

    Nets= []
    nets = []
    input_dims = []
    input_dim = []
    for i in select_nets:
        Net = np.loadtxt('../PPMI/' + 'PPMI_' + str(i) + '.txt')
        Nets.append(minmax_scale(Net))
        input_dims.append(Net.shape[1])

    for i in select_net:
        net= np.loadtxt('../Sim_data/' + 'sim_' + str(i) + '.txt')
        nets.append(minmax_scale(net))
        input_dim.append(net.shape[1])


    model_names = []
    for a in select_arch:
        print ("### [%s] Running for architecture: %s" % (model_type, str(arch[model_type][a])))
        mid_model, history = build_model(Nets, input_dims, arch[model_type][a], nf, 1.0, model_type, epochs, batch_size)

        features = mid_model.predict(Nets)
        features = minmax_scale(features)
        print(features.shape[1])
        np.savetxt('../MDAfeature/PPMIfeatures.txt', features)

        mid_model, history = build_model(nets, input_dim, arch[model_type][a], nf, 1.0, model_type, epochs, batch_size)

        features = mid_model.predict(nets)
        features = minmax_scale(features)
        print(features.shape[1])
        np.savetxt('../MDAfeature/Simfeatures.txt', features)






