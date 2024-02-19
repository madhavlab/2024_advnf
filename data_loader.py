import pickle
import gzip
import numpy as np

def load_data_mh_generated(filename):
    
    f = open(filename, 'rb')
    if (f.read(2) == '\x1f\x8b'):
        f.seek(0)
        return gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
    
    training_inputs = pickle.load(f,encoding="latin1")
    lattice_width   = len(training_inputs[0][0])
    lattice_height  = len(training_inputs[0][1])
    training_inputs = np.reshape(training_inputs,(-1,lattice_width,lattice_height,1))
    f.close()
    return training_inputs


def load_data_model_generated(filename):
    f = open(filename, 'rb')
    if (f.read(2) == '\x1f\x8b'):
        f.seek(0)
        return gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
    
    training_inputs = pickle.load(f,encoding="latin1")
    lattice_width   = (training_inputs[0]).shape[1]
    lattice_height  = (training_inputs[0]).shape[2]
    training_inputs = np.reshape(training_inputs,(-1,lattice_width,lattice_height,1))
    f.close()
    return training_inputs
    