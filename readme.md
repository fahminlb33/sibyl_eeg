# Sibyl EEG: Deep Learning for EEG classification

This repo contains source code used in [this research](http://dx.doi.org/10.13140/RG.2.2.19024.84486) to classify EEG dataset to identify normal (control) EEG pattern and alcoholic EEG pattern. This repo uses Python 3.8 and TensorFlow 2.3 to perform the classification task. You can find more about the used libraries in the `requirements.txt` file.

If you want to run this code locally, please create a new Anaconda environment and then install the required libraries using `conda create --name sibyl_eeg --file requirements.txt` or using `pip install -r requirements.txt`.

## Dataset tools

You can download, transform, and compress the dataset using the `dataset.py`.

```bash
# download the full dataset into current directory
python dataset.py download full .

# transform the downloaded dataset into CSV
python dataset.py transform eeg_full.tar dataset.csv

# compress the CSV dataset into parquet, if necessary
python dataset.py compress dataset.csv dataset_compressed.parquet
```

## Visualizing data

Visualize EEG dataset into five different visualizations.

```bash
# list all available channels in the dataset
python visualize.py channels dataset.csv

# visualize EEG recording
python visualize.py signal dataset.csv --id a_1_co2a0000364 --trial 0 --channel AF1,F7

# visualize EEG recording with FFT
python visualize.py signal_fft dataset.csv --id a_1_co2a0000364 --trial 0 --channel AF1

# visualize single trial from a subject as topograph
python visualize.py topograph dataset.csv --id a_1_co2a0000364 --trial 0

# animated topograph
python visualize.py topograph_animate dataset.csv --id a_1_co2a0000364 --trial 0
```

## Training the neural network

Run neural network training with customized settings and parameters.

```bash
# run training with model Dense-128
python train.py dense dataset.csv --units 128 --epochs 20 --evaluation-split 0.30 --validation-split 0.20

# run training with model CNN-128,64
python train.py cnn dataset.csv --units 128,64 --epochs 20 --evaluation-split 0.30 --validation-split 0.20

# run training with model CNN-NR-128,64
python train.py cnn dataset.csv --units 128,64 --normalize --dropout 0.25 --epochs 20 --evaluation-split 0.30 --validation-split 0.20

# run training with model LSTM-128,64
python train.py lstm dataset.csv --units 128,64 --epochs 20 --evaluation-split 0.30 --validation-split 0.20
```

You can always contribute to this research by creating issues, pull request, or commenting on the research on ResearchGate.

## References

1. http://dx.doi.org/10.13140/RG.2.2.19024.84486
