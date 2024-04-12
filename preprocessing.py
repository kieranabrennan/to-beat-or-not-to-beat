from scipy.signal import firwin, lfilter
import samplerate
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

def lowpass_fir_record(record, fs, order, cutoff):
    '''
    Applies a finite impulse response low pass filter to a 1d signal
    '''
    coef = firwin(order+1, cutoff, fs=fs)
    return lfilter(coef, 1.0, record)

def resample_record(record, fs, resample_fs):
    '''
    Resamples a 1d signal with a specificed resample rate
    '''
    fs_ratio = resample_fs / fs
    return samplerate.resample(record, fs_ratio)

def normalise_record(record, mean, std):
    '''
    Normalises a signal
    '''
    return (record - mean) / std

def lowpass_filter_and_resample_record_list(record_list, fs_list, order=512, cutoff=60, resample_fs=120):
    '''
    Applies a low-pass filter and resamples all records (numpy array) in a list
    '''
    resampled_records = []
    for record, fs in tqdm(zip(record_list, fs_list), desc='Resampling records', total=len(record_list)):
        signal = lowpass_fir_record(record, fs, order, cutoff)
        signal = resample_record(signal, fs, resample_fs)
        resampled_records.append(signal)
    return resampled_records

def normalise_record_list(records, mean=None, std=None):
    '''
    Normalises all 1d signals in a list based on the mean and std of the dataset (optional inputs)
    '''
    # mean and std are across entire dataset
    records_all = np.concatenate(records)
    if mean is None and std is None:
        mean = np.mean(records_all)
        std = np.std(records_all)
    normalised_records = []
    for record in records:
        record = normalise_record(record, mean, std)
        normalised_records.append(record)
    return normalised_records

def drop_other_class_records_and_labels(records, labels):
    af_and_normal_labels = labels[labels['A'] | labels['N']].copy()
    af_and_normal_labels.drop(['O','~'], axis=1, inplace=True)
    af_and_normal_ids = af_and_normal_labels.index.to_list()
    af_and_normal_records = [records[id] for id in af_and_normal_ids]
    
    return af_and_normal_records, af_and_normal_labels.reset_index(drop=True)

def get_afib_records(records, labels):
    '''
    Returns atrial fibrillation records from list of numpy array (records),
    where labels is a dataframe with boolean column "A" indicating afib record
    '''
    af_labels = labels[labels['A']]
    af_ids = af_labels.index.to_list()
    af_records = [records[id] for id in af_ids]
    return af_records, af_labels.reset_index(drop=True)

def duplicate_records(records, labels, duplication_factor=3):
    '''
    Duplicates records (list of numpy array) and labels (dataframe) by the duplication factor
    '''
    records = records * duplication_factor
    labels = pd.concat([labels] * duplication_factor, ignore_index=True)
    return records, labels

def duplicate_afib_records_in_list(records, labels, dup_factor):
    af_records, af_labels = get_afib_records(records, labels)
    af_records, af_labels = duplicate_records(af_records, af_labels, dup_factor)

    dup_records = records + af_records
    dup_labels = pd.concat([labels, af_labels], ignore_index=True)
    return dup_records, dup_labels

def crop_random_and_pad_sample(sample, fs, crop_length):
    '''
    Crops a sample randomly by crop_length (seconds), where fs is the sample rate in Hz.
    Returns the full sample if the crop length is greater than the sample length
    '''
    n_crop = int(crop_length*fs)+1
    if n_crop > len(sample): # Post-pad with zeros
        pad_size = n_crop - len(sample)
        sample = np.pad(sample, (0,pad_size))
        return sample
    else: # Randomly cropping
        max_start_index = max(len(sample)-n_crop, 0)
        start_index = random.randint(0,max_start_index)
        crop_sample = sample[start_index:(start_index+n_crop)]
        return crop_sample
    
def crop_and_pad_record_list(records, fs, crop_length):
    cropped_records = []
    for record in records:
        crop_record = crop_random_and_pad_sample(record, fs, crop_length)
        cropped_records.append(crop_record)
    return cropped_records
