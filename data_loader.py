import os

import scipy.io as sio

from bnsNMF.datatools import nan_present

def data_loader(data_type):

    if data_type == 'eeg':
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                    "data","EEG_PhysioNet",
                                    "SpecsAndLabels.mat"))

        specs_labels = sio.loadmat(data_dir)

        # each variable below is a list containing data from all 19 subjects
        night_one_eeg = specs_labels['SPEC_1'][0]
        night_one_labels_4 = specs_labels['ANNOT_1'][0] # using 4 labels
        night_one_labels_6 = specs_labels['ANNOTORIG_1'][0] # using 6 labels
        night_two_eeg = specs_labels['SPEC_2'][0]
        night_two_labels_4 = specs_labels['ANNOT_2'][0] # using 4 labels
        night_two_labels_6 = specs_labels['ANNOTORIG_2'][0] # using 6 labels

        for X in night_one_eeg:
            print(nan_present(X))
        return X
