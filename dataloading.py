from pathlib import Path
import pandas as pd
from tqdm import tqdm
import wfdb
import pickle
import pyedflib
import numpy as np

def read_challenge17_data(db_dir):
    """
    Read the 2017 PhysioNet Computing in Cardiology (CinC) Challenge data set given the directory (db_dir)
    Returns and list of records (1D numpy arrays), 1D array of sample frequencies, a dataframe of labels one hot encoded labels).
    """
    db_dir = Path(db_dir)
    if not db_dir.is_dir():
        raise ValueError('Provided path is not a directory: %s' % db_dir)
    index_file = db_dir / 'RECORDS'
    reference_file = db_dir / 'REFERENCE.csv'

    if not index_file.is_file():
        raise ValueError('Index file does not exist')
    if not reference_file.is_file():
        raise ValueError('Reference file does not exist')
    records_index = pd.read_csv(index_file, names=['record_name'], dtype='str', index_col='record_name')
    references = pd.read_csv(reference_file, names=['record_name', 'label'], index_col='record_name', dtype='str')
    
    references = pd.merge(records_index, references, on='record_name')
    label_df = pd.get_dummies(references.label)
    records_iterator = references.iterrows()
    records_iterator = tqdm(records_iterator, total=len(references), desc='Reading records')
    records = []
    for record_name, _ in records_iterator:
        record_file = db_dir / record_name
        record = wfdb.rdrecord(str(record_file))
        records.append(record)
    record_list = [record.p_signal.flatten() for record in records]
    fs_list = [record.fs for record in records]
    label_df = label_df.reset_index(drop=True)
    return record_list, fs_list, label_df

def save_challenge17_pkl(save_path, data):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_challenge17_pkl(save_path):
    with open(save_path, 'rb') as f:
        return pickle.load(f)

def read_ecg_from_edf(file_path):
    '''
    Reads an ecg signal from an edf file given the file path
    EDF is European Data Format, a standard format for medical time series, and exportable from the Polar H10 ECG Analysis App
    Returns a tuple of t and signal both as numpy arrays. t is derived from the sample rate
    '''

    with pyedflib.EdfReader(file_path) as edf:
        # Get the number of signals in the file
        n = edf.signals_in_file
        
        #  Assuming one signal
        ecg = edf.readSignal(0)
        fs = edf.getSampleFrequency(0)
        print(f"EDF file sample rate: {fs}")

        dt = 1/fs
        t = np.arange(0,dt*(len(ecg)), dt)
        signal_ids = t > 5 # Remove first 5 seconds
        ecg = ecg[signal_ids]
        t = t[signal_ids]
        t = t - t[0]

    return (t, ecg)