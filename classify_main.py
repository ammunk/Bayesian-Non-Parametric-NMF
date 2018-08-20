import sys
import time
import os
import argparse
import pickle
from pathlib import Path
import scipy.io as sio
import numpy as np
from sacred import Experiment

# create experiment, using the data_loader ingredient
ex = Experiment('Classify')
@ex.named_config
def malNMF():
    model = "malNMF"
    experiment_path = Path(__file__).parent / "results" / "classifiers" \
                                            / "learned_representations" \
                                            / "malNMF"

@ex.named_config
def malPCA():
    model = "malPCA"
    experiment_path = Path(__file__).parent / "results" / "classifiers" \
                                            / "learned_representations" \
                                            / "malPCA"

@ex.named_config
def gibbsvb():
    model = "gibbsvb"
    experiment_path = Path(__file__).parent / "cluster_directory" \
                                            / "cluster_results" / "EEG"

@ex.named_config
def gibbs():
    model = "gibbs"
    experiment_path = Path(__file__).parent / "cluster_directory" \
                                            / "cluster_results" / "EEG"

@ex.named_config
def vb():
    model = "vb"
    experiment_path = Path(__file__).parent / "cluster_directory" \
                                            / "cluster_results" / "EEG"

@ex.named_config
def vb_ibp():
    model = "vb_ibp"
    experiment_path = Path(__file__).parent / "cluster_directory" \
                                            / "cluster_results" / "EEG"

@ex.config
def cfg():
    experiment_path = Path(__file__).parent / "results" / "classifiers" \
                                            / "learned_representations" \
                                            / "malPCA"
    subject = 0
    result_path = experiment_path.parent / "classifier_results"
    model = "malPCA"
    features_to_use = 1
    W_hyperprior = "shared"
    K = 16 # Possibilities: 16, 25, 50, 80, 100

@ex.capture
def get_data(subject):
    data_path = Path(__file__).parent / "data" / "EEG_PhysioNet"\
                                  / "SpecsAndLabels.mat"
    specs_labels = sio.loadmat(data_path)

    # each variable below is a list containing data from all 19 subjects
    night_one_eeg = specs_labels['SPEC_1'][0]
    night_one_labels_4 = specs_labels['ANNOT_1'][0] # using 4 labels
    #night_one_labels_6 = specs_labels['ANNOTORIG_1'][0] # using 6 labels
    night_two_eeg = specs_labels['SPEC_2'][0]
    night_two_labels_4 = specs_labels['ANNOT_2'][0] # using 4 labels
    #night_two_labels_6 = specs_labels['ANNOTORIG_2'][0] # using 6 labels

    Xtrain = night_one_eeg[subject].T
    Xtest = night_two_eeg[subject].T
    Ytrain = night_one_labels_4[subject]
    Ytest = night_two_labels_4[subject]
    _, divisor = Xtrain.shape
    X = np.append(Xtrain, Xtest, axis=1)
    return Ytrain, Ytest, divisor

@ex.capture
def data_selector(model, experiment_path, subject, W_hyperprior, K):

    def get_ids(path):
        ids = []
        for i, filename in enumerate(path.iterdir()):
            ids.append(filename.stem.split('_')[-1])
            if i == 9:
                break
        return ids

    dat_file = Path("model_configuration_" + model + ".pickle")
    if "mal" not in model:
        path = experiment_path / ("subject_" + str(subject)) / model\
                               / W_hyperprior
        ids = get_ids(path)
        path = path / f"factors_{ids[0]}.pickle"

    elif model == "malNMF":
        path = experiment_path / str(K) / ("subject_" + str(subject)) / str(1)\
                               / dat_file
    else:
        path = experiment_path / ("subject_"+str(subject)) / str(1) / dat_file

    with open(path, 'br') as f:
        conf = pickle.load(f)
        if model in ['vb_ibp', 'gibbs', 'gibbsvb']:
            return conf['W']*conf['Z']
        else:
            return conf['W']

@ex.automain
def main(model, features_to_use, K, W_hyperprior, subject, _run, _log):
    from sklearn.linear_model import LogisticRegression as LR

    W = data_selector()
    DIM = W.shape[0]
    Ytrain, Ytest, divisor = get_data()
    if model == "malPCA":
        W = W[:features_to_use,:]

    Xtrain, Xtest = W[:,:divisor].T, W[:,divisor:].T # transpose to comply to
                                                     # sklearn standards

    msg = f"Classifying EEG\n===================================="
    msg += f"\n\t-Subject: {subject}"
    msg +=(f"\n\t-MODEL TYPE: {model}")
    if model == "malNMF":
        msg += f"\n\t-K: {K}"
    elif model == "malPCA":
        msg += f"\n\t-Chosen features: {features_to_use}"
    _log.info(msg)
    lr = LR(multi_class='multinomial', solver='newton-cg')
    lr.fit(Xtrain, Ytrain.ravel())
    score = lr.score(Xtest, Ytest.ravel())
    results_file = Path(__file__).parent / "results" / "classifiers"\
                                         / "classifier_results" / "EEG"\
                                         / f"subject_{subject}.txt"
    with open(results_file, "a") as f:
        if model == "malNMF":
            text = f"{model}_{K}\t\t{score}\t\t{DIM}"
        elif model == "malPCA":
            text = f"{model}_{features_to_use}\t\t{score}\t\t{DIM}"
        else:
            text = f"{model}_{W_hyperprior}\t\t{score}\t\t{DIM}"
        print(text, file=f)
