import os
import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler


class MindBigData(Dataset):
    """MindBigData dataset."""
    def __init__(self, inputs, labels, transform=None):
        """
        Args:
            inputs (2D np array): Contains EEG signals from different channels
            labels (np array): digit seen by patient
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels = labels
        self.inputs = inputs
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        label = self.labels[i]
        input = self.inputs[i]

        if self.transform:
            return self.transform(label), self.transform(input)

        return input, label


class EEGImagesDataset(Dataset):
    """EEG Images Dataset from EEG."""
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def GetDataSet(input_file, num_samples=-1, samples_per_digit=2000):
    """
    Read data from MindBigData - The Visual "MNIST" of Brain Digits (2021)
    for more details refer: https://www.mindbigdata.com/opendb/visualmnist.html
    :param input_file: input file to read data from
    :param num_samples: number of samples(lines) to read from the file
    :param samples_per_digit: number of samples per digit
    :return: x, y, labels_hist
    """
    x = []
    y = []
    labels_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cnt = 0
    num_channels = 14
    channels_cnt = 0
    all_channels = []
    with open(input_file) as data:
        csv_reader = csv.reader(data, delimiter='\t')

        for row in csv_reader:
            if (num_samples == -1):
                pass
            elif(cnt == num_samples):
                break

            cnt += 1

            id = row[0]  # a numeric, only for reference purposes.

            event = row[1]  # an integer, used to distinguish the same event captured at different brain locations

            device = row[2]  # 2 character string, to identify the device used to capture the signals

            channel = row[3]  # a string, to identify the 10/20 brain location of the signal,
            # with possible values: "AF3,"AF4","T7","T8","PZ"

            code = int(row[4])  # the digit been thought/seen, with possible values 0,1,2,3,4,5,6,7,8,9 or -1
            if (code == -1):
                continue

            size = int(row[5])  # size = number of values captured in the 2 seconds of this signal
            if (size < 256):
                continue

            data = (row[6]).split(',')  # a coma separated set of numbers, with the time-series amplitude of the signal
            data = np.array(data, dtype=np.float32)
            data = data[0:256]

            # define a number of samples per digit
            if (labels_hist[code] >= samples_per_digit):
                continue

            all_channels.append(data)
            channels_cnt += 1
            bad_sample = False
            if (channels_cnt == num_channels):
                for i in range(len(all_channels)):
                    # Standardization
                    s_mean = np.mean(all_channels[i])
                    s_std = np.std(all_channels[i])
                    if (s_std == 0):
                        bad_sample = True
                        break
                    data_std = (all_channels[i] - s_mean) / s_std

                    # min-max scaling
                    s_min = np.min(data_std)
                    s_max = np.max(data_std)
                    if (s_max - s_min == 0):
                        bad_sample = True
                        break
                    all_channels[i] = (data_std - s_min) / (s_max - s_min)
                if(not(bad_sample)):
                    x.append(np.array(all_channels))
                    y.append(code)
                    labels_hist[code] += 1
                all_channels = []
                channels_cnt = 0

    return np.array(x), np.array(y).astype(np.int), labels_hist


def GetDataAndPreProcess(input_file, num_samples=-1, samples_per_digit=5000):
    """
    get data set and perform pre-processing
    :param input_file: input file to read data from
    :param num_samples: number of samples(lines) to read from the file
    :param samples_per_digit: number of samples per digit
    :return: x, y
    """
    # Read train and test datasets from file
    x, y, labels_hist = GetDataSet(input_file=input_file, num_samples=num_samples, samples_per_digit=samples_per_digit)
    print(labels_hist)

    # Pre-Process data
    x = PreProcess(x)

    return x, y


def GetDataLoaders(x, y, batch_size=64):
    """
    get data loaders
    :param x: pre-processed EEG data
    :param y: labels
    :param batch_size: batch size
    :return: train_loader, valid_loader, test_loader
    """
    x_new = np.copy(x)
    y_new = np.copy(y)

    # Split data: 75% train, 25% test
    x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.25)

    # create DataSet object
    train_data = MindBigData(x_train, y_train)
    test_data = MindBigData(x_test, y_test)

    # Preparing for validation set
    indices = list(range(len(test_data)))
    np.random.shuffle(indices)

    # get 60% of the test set for validation
    split = int(np.floor(0.6 * len(test_data)))
    valid_sample = SubsetRandomSampler(indices[:split])
    test_sample = SubsetRandomSampler(indices[split:])

    # Create DataLoaders
    test_loader = DataLoader(test_data, sampler=test_sample, batch_size=batch_size)
    valid_loader = DataLoader(test_data, sampler=valid_sample, batch_size=batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def CreateSpectrograms(data):
    """
    create spectrogram's of EEG signals
    :param data: dataset of EEG signals of shape (Nx250)
    :return: data_new
    """
    data_new = []
    for i in range(data.shape[0]):
        # spectrogram, _, _, img = specgram(data[i], NFFT=127, Fs=128, noverlap=125)
        f, t, spectrogram = signal.stft(data[i,32:256], fs=128, nperseg=128, noverlap=124, detrend='constant')
        spectrogram = abs(spectrogram)
        # min-max scaling
        s_min = np.min(spectrogram)
        s_max = np.max(spectrogram)
        spectrogram = (spectrogram - s_min) / (s_max - s_min)
        # reduce size to 64x64
        spectrogram = spectrogram[0:64, 0:64]
        # convert to range 0-255
        spectrogram = (spectrogram*255).astype(np.uint8)
        data_new.append(spectrogram.astype(np.uint8))

    return data_new


def SaveDataset(dir_path, data, labels):
    """
    save images and labels
    :param dir_path: path to dataset directory (train/valid/test)
    :param data: images
    :param labels: list of labels
    """
    img_path = os.path.join(dir_path, 'img')
    labels_file = os.path.join(dir_path, 'labels.csv')
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    rows = []
    # save images
    for i in range(len(data)):
        cv2.imwrite(os.path.join(img_path, 'img{}.png'.format(i)), data[i])

        # save labels in csv file
        rows.append(['img{}.png'.format(i), labels[i]])

        with open(labels_file, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(rows)


def GetDataLoadersEEGImages(input_file, num_samples=-1, batch_size=64, gen=True, samples_per_digit=2000):
    """
    generate data loaders of spectrogram's of EEG signals with the specified batch_size
    :param input_file: input file to read EEG signals data from
    :param num_samples: number of samples(lines) to read from the file
    :param batch_size: batch_size
    :param gen: when True: generate images. when False: load pre-created images
    :param samples_per_digit: number of samples per digit
    :return: train_loader, test_loader
    """
    if (gen is True):  # load dataset and create images
        # Read train and test datasets from file
        x, y, labels_hist = GetDataSet(input_file=input_file, num_samples=num_samples,
                                       samples_per_digit=samples_per_digit)
        print(labels_hist)

        # Pre-Process data
        x = PreProcess(x)

        # Split data: 80% train, 20% test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # Create spectrogram
        x_train_new = CreateSpectrograms(x_train)
        x_test_new = CreateSpectrograms(x_test)

        # save images and labels
        print("save dataset")
        SaveDataset(dir_path='sample_data/train', data=x_train_new, labels=y_train)
        SaveDataset(dir_path='sample_data/test', data=x_test_new, labels=y_test)

    # load pre-created images
    # create DataSet object
    train_data = EEGImagesDataset('sample_data/train/labels.csv', 'sample_data/train/img',
                                  transform=transforms.ToTensor())
    test_data = EEGImagesDataset('sample_data/test/labels.csv', 'sample_data/test/img',
                                 transform=transforms.ToTensor())

    # Preparing for validation set
    indices = list(range(len(test_data)))
    np.random.shuffle(indices)

    # get 50% of the test set for validation
    split = int(np.floor(0.5 * len(test_data)))
    valid_sample = SubsetRandomSampler(indices[:split])
    test_sample = SubsetRandomSampler(indices[split:])

    # Create DataLoaders
    test_loader = DataLoader(test_data, sampler=test_sample, batch_size=batch_size)
    valid_loader = DataLoader(test_data, sampler=valid_sample, batch_size=batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def PreProcess(x):
    """
    Apply pre-processing to EEG signals such as filtering and noise reduction
    :param x
    :return: x_new
    """
    fs = 128.0  # Sample frequency (Hz)
    x_new = np.copy(x)
    x_trimmed = np.zeros((x.shape[0], x.shape[1], 224), dtype=np.float32)

    # Design butterworth filter
    b_butter1, a_butter1 = signal.butter(N=6, Wn=[0.5, 63], btype='bandpass', fs=fs)

    # Design notch filter
    b_notch, a_notch = signal.iirnotch(w0=50, Q=30, fs=fs)

    for i in range(x.shape[0]):
        # apply filter 1
        x_new[i] = signal.lfilter(b_butter1, a_butter1, x[i])

        # apply notch filter
        x_new[i] = signal.lfilter(b_notch, a_notch, x_new[i])

        # trim 32 samples from the beginning of the signal
        x_trimmed[i] = x_new[i][:, 32:256]

    return x_trimmed
