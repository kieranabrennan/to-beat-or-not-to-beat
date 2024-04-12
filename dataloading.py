from pathlib import Path
import pandas as pd
from tqdm import tqdm
import wfdb
import pickle

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
