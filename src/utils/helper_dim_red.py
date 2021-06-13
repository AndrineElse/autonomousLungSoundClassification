from os import path
module_path = path.abspath(path.join('../..'))

from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA 
from sklearn.manifold import TSNE
import random
import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

font = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 11, weight = 1000)
font_small = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 8, weight = 1000)


def get_dim_reduced_X(X, method, X_train = None, y_train = None):
    if method == 'ica':
        ica = FastICA(n_components= 2)
        X_new = ica.fit_transform(X)
    
    ## PCA 
    elif method == 'pca':
        pca = PCA(n_components=2)
        X_new = pca.fit_transform(X) 
        
    ## Autoencoder
    elif method == 'autoencoder':
        encoder = get_trained_encoder(X_train,y_train, 2 )
        X_new = encoder.predict(X)
        
    ## TSNE   
    elif method == 'tsne':
        X_new = TSNE(n_components= 2, random_state = 0).fit_transform(X)
    else:
        print('Need to specify a dimentionality reduction technique')
        return None
    return X_new

def get_trained_encoder(X_train, y_train, red_dim = 10):
    '''
    Returns the trained encoder part of a autoencoder
    '''
    encoding_dim = red_dim
    input_layer = Input(shape=(X_train.shape[1],))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)

    # let's create and compile the autoencoder
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    X1, X2, Y1, Y2 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    Y1 = np.asarray(Y1).astype('float32').reshape((-1,1))
    Y2 = np.asarray(Y2).astype('float32').reshape((-1,1))


    autoencoder.fit(X1, Y1,
                    epochs=300,
                    batch_size=200,
                    shuffle=False,
                    verbose = 0,
                    validation_data=(X2, Y2))
    
    
    encoder = Model(input_layer, encoded)
    return encoder


def scatterplot_with_colors(X, y, module_path = module_path, new_legends = None, x_y_labels = None):
    font = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 11, weight = 1000)
    font_small = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 8, weight = 1000)
    x_label = 'x'
    y_label = 'y'
    if x_y_labels:
        x_label = x_y_labels[0]
        y_label = x_y_labels[1]
        
    df = pd.DataFrame({x_label : X[:,0], 
                       y_label : X[:,1],
                       'label' : y})
    
    
    colors = ['#F94144', '#90BE6D', '#577590','#F3722C', '#F8961E', '#F9844A', '#F9C74F', '#43AA8B', '#4D908E', '#277DA1']
    colors = ['#F9414466', '#90BE6D66', '#57759066','#F3722C66', '#F8961E66',
              '#F9844A66', '#F9C74F66', '#43AA8B66', '#4D908E66', '#277DA166']

    
    colorsDict = {idx : color for (idx, color) in enumerate(colors)}
    colorsDict[-1] = '#484848'
    f, ax = plt.subplots(1,1) # 1 x 1 array , can also be any other size
    f.set_size_inches(3, 3)
    
        
    grouped = df.groupby('label')
    
    for key, group in grouped:
        label_key = key
        if key == -1:
            label_key = 'deleted samples'
        ax = group.plot(ax=ax, kind='scatter', x=x_label,y=y_label,
                        label=label_key, color=colorsDict[key] ,
                        s = 3, alpha=0.8)


    for label in ax.get_ylabel() :
        ax.set_ylabel(y_label, fontproperties = font)
    for label in ax.get_xlabel() :
        ax.set_xlabel(x_label, fontproperties = font)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(font)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(font)
    
    #ax.legend(prop=font)
    

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, new_legends, prop = font)
        
    return f, ax



def add_noise_dataset(X, ampl = 10, noise_amount = 4):
    '''
    Noise amount is inverse, meaning higher noise amout, smaller fraction of samples will be effected by noise
    '''
    noise_indices = np.random.RandomState(seed=1).permutation(X.index.tolist())[0:len(X)//noise_amount]
    new_X = X.copy()
    for [idx, row] in new_X.iterrows():
        if idx in noise_indices:
            sig = row
            noise = np.random.RandomState(seed=idx%10).normal(0, ampl, len(sig))
            new_X.iloc[idx] = pd.Series((sig + noise).tolist())
    return new_X , noise_indices




def get_auc_scores(X_test, y_test, clf_dict, indices):
    auc_curve_svm ={}
    auc_curve_init = {}
    removed_data = {}
    for name, idx in indices.items():
        pred = clf_dict[name].predict(X_test.iloc[idx])
        fpr, tpr, _ = metrics.roc_curve(y_test.iloc[idx], pred)
        auc_curve_svm[name] = metrics.auc(fpr, tpr)

        pred = clf_dict[name].predict(X_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, pred)
        auc_curve_init[name] = metrics.auc(fpr, tpr)
        
        
        removed_data[name] = len(idx)/len(X_test)
    return auc_curve_init , auc_curve_svm, removed_data