from dataloading import *
from preprocessing import *
from utils import *
from modelling import *
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

db_dir = "data/physionet"
pkl_path = db_dir + "normalisedrecords_fslist_labels.pkl"
cutoff = 60 # Hz
resample_fs = 120 # Hz
crop_length = 30 # s
afib_dup_factor = 3

if not os.path.exists(pkl_path):
    # Read filter, and normalise
    record_list, fs_list, labels = read_challenge17_data(db_dir)
    resampled_records = lowpass_filter_and_resample_record_list(record_list, fs_list, 512, cutoff, resample_fs)
    normalised_records = normalise_record_list(resampled_records)
    # Save it out 
    save_challenge17_pkl(pkl_path, (normalised_records, fs_list, labels))
else:
    # Read in the pkl file
    normalised_records, fs_list, labels = load_challenge17_pkl(pkl_path)

normalised_records, labels = drop_other_class_records_and_labels(normalised_records, labels)
dup_records, dup_labels = duplicate_afib_records_in_list(normalised_records, labels, afib_dup_factor)
cropped_records = crop_and_pad_record_list(dup_records, resample_fs, crop_length)


BATCH_SIZE = 64
EPOCHS = 400
K_FOLDS = 10
STREAM2_SIZE = 9

X = np.array(cropped_records)
y = np.array(dup_labels)
y = y[:,0].astype(int)
X = np.expand_dims(X, -1)

kf = KFold(n_splits=K_FOLDS, shuffle=True)

test_scores = []
fold_id = 0
train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = 'models/' + train_time 
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    logs_dir = 'logs/' + train_time + f'/{fold_id}'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    model = create_dual_stream_cnn_model((X_train.shape[1], 1), stream2_size = STREAM2_SIZE)
    print_gpu_availability()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, mode='min', restore_best_weights=True, verbose=1)
    model.fit(train_dataset, validation_data = validation_dataset,
            epochs=EPOCHS, verbose=1,
            callbacks=[lr_scheduler, tensorboard_callback, early_stopping_callback])
    
    models_path = models_dir + f'/fold_{fold_id}_model_weights.h5'
    model.save_weights(models_path)

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)
    y_scores = model.predict(X_test, verbose=0)
    y_scores = y_scores.flatten()
    test_fpr, test_tpr, _ = roc_curve(y_test, y_scores)
    test_auc = roc_auc_score(y_test, y_scores)

    test_scores.append({'loss':test_loss, 
                        'acc': test_accuracy, 
                        'prec':test_precision, 
                        'rec':test_recall, 
                        'auc':test_auc, 
                        'fpr':test_fpr.tolist(), 
                        'tpr':test_tpr.tolist()})

    fold_id += 1

test_scores_path = models_dir + "/test_scores.json"
with open(test_scores_path, "w") as file:
    json.dump(test_scores, file, indent=4)

kfold_accuracy = np.array([elem['acc'] for elem in test_scores])
kfold_precision = np.array([elem['prec'] for elem in test_scores])
kfold_recall = np.array([elem['rec'] for elem in test_scores])
print(f"Accuracy:\t{100*kfold_accuracy.mean():.1f}% \nPrecision:\t{100*kfold_precision.mean():.1f}% \nRecall:\t\t{100*kfold_recall.mean():.1f}%")

tpr = test_scores[0]['tpr']
fpr = test_scores[0]['fpr']
auc = test_scores[0]['auc']

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--', color='black')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f"AUC: {auc:.4f}")
plt.savefig(models_dir+'/roc_curve.png')  # Save the plot as a PNG file