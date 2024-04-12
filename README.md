# To Beat Or Not To Beat — Machine Learning Atrial Fibrillation Detection 

Atrial Fibrillation (AF) is the most common, serious type of heartbeat arrthymia, affecting around 30 million people worldwide. It occurs when there are irregularities in the electrical wave the generates a heartbeat, creating misfirings of the upper, atrial chambers which receive blood from the body and lungs. Regular blood transport is disrupted creating vortices which reduce the efficiency of the heart and can create clots. People with AF have a 4 times higher risk of mortality and 5 times higher risk of stroke than the normal population. 

This program is an implementation of the paper Multiscaled fusion of deep convolutional neural networks for screening atrial fibrillation from single lead short ECG recordings. IEEE journal of biomedical and health informatics, 22(6), 1744-1753, Fan, X., Yao, Q., Cai, Y., Miao, F., Sun, F., & Li, Y. (2018). A convolutional neural network is trained as a binary classifier to detect cases of atrial fibrillation using the Physionet Computing in Cardiology Challenge 2017 (https://physionet.org/content/challenge-2017/1.0.0/).


## Setup

``` pip install -r requirements.txt 
    python3 trainer.py
```

Notes:

To get GPU working had to run
pip install tensorflow[and-cuda]==2.15.0.post1
Running just pip install installed 2.16 which could not recognize the GPU
