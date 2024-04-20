# To Beat Or Not To Beat — Machine Learning Atrial Fibrillation Detection 

Atrial Fibrillation (AF) is the most common serious heart arrthymia, affecting around 30 million people worldwide. It occurs when there are irregularities in the electrical wave the generates a heartbeat, creating misfirings of the upper, atrial chambers which receive blood from the body and lungs. Regular blood transport is disrupted creating vortices which reduce the efficiency of the heart and can create clots. People with AF have a 4 times higher risk of mortality and 5 times higher risk of stroke than the normal population. With this program and a Polar H10 Heart rate monitor you can assess your risk of atrial fibrillation from a 60 second recording.

This is an implementation of the paper Multiscaled fusion of deep convolutional neural networks for screening atrial fibrillation from single lead short ECG recordings. IEEE journal of biomedical and health informatics, 22(6), 1744-1753, Fan, X., Yao, Q., Cai, Y., Miao, F., Sun, F., & Li, Y. (2018). A convolutional neural network is trained as a binary classifier to detect cases of atrial fibrillation using the Physionet Computing in Cardiology Challenge 2017 (https://physionet.org/content/challenge-2017/1.0.0/).

<p align="center">
  <img src="./img/sample_af.png" width="48%" />
  <img src="./img/sample_n.png" width="48%" />
</p>

## Setup & Usage

### Inference
There are three options for inference using a pre-trained model in Inference.ipynb:
1. Run a Atrial Fibrillation prediction immediately by connecting to a Polar H10 HR monitor.
2. Run a prediction on pre-recorded ECG data loaded from an .edf file.
3. Run a prediction on a sample from the Physionet dataset, select a random sample to run inference

#### 1. Polar H10 Inference
- Connect the Polar H10 chest strap
- Run the first cell in Inference.ipynb, which will connect to the Polar H10 via Bluetooth and take a 60 second ECG recording, and preprocesses the data
- Run the last cell, which loads the model and runs inference to display the results

#### 2. Pre-recorded ECG Inference
- Uploaded a pre-recorded ECG .edf file into the data directory. This can be recorded for example with the "Polar H10 ECG Analysis App" on the Google Play Store
- Run the second cell in Inference.ipynb, which loads this file and preprocesses the data
- Run the last cell, to run inference

#### 3. Physionet dataset
- Run the third cell in Inference.ipynb, which loads a random sample from the Physionet database and preprocesses the data
- Run the last cell to make a prediction

### Training
To train the network

``` 
    pip install -r requirements.txt 
    python3 trainer.py
```

Notes:
To get GPU working had to run
pip install tensorflow[and-cuda]==2.15.0.post1
Running just pip install installed 2.16 which could not recognize the GPU

For the samplerate package, had to use
pip -q install git+https://github.com/tuxu/python-samplerate.git@fix_cmake_dep

## Benchmark
10-fold test results for ecg signals of 30 s duration (original paper results in parentheses)

Model in this repository uses a binary classification signal sigmoidal activation and binary cross entropy loss

| Model   | Sensitivity | Precision   |  Accuracy   |
| :------ | :---------: | :---------: | :---------: |
| (3,3)   |98.4% (85.9%)|96.6% (92.0%)|98.1% (97.3%)|
| (3,5)   |98.3% (89.2%)|**97.1%** (84.1%)|98.3% (96.5%)|
| (3,7)   |98.7% (89.9%)|96.8% (91.4%)|98.3% (97.8%)|
| (3,9)   |**98.7%** (88.6%)|96.9% (91.1%)|**98.4%** (97.4%)|

## Disclaimer
This tool is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Users are advised to seek the advice of qualified health providers with any questions regarding a medical condition.

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
