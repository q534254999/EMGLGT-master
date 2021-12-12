# import matplotlib.pyplot as plt
import os
import time

import numpy as np
import re
import csv
# import os
# import tensorflow as tf
# import pandas as pd
import math
# import librosa
# import librosa.display
# import neurokit2 as nk
import pandas as pd
# from matplotlib import pyplot as plt
from scipy import integrate
from scipy.signal import butter, lfilter
from fcn_model import get_model

# 记录时长
RecordPeriod = 60
WINSIZE = 10
SAMPLEFZ = 96

def max_emg(data):
    return np.max(data)
def min_emg(data):
    return np.min(data)
def mean_emg(data):
    return np.mean(data)
def variance(data):
    return np.average((data-np.mean(data))**2)
#频率为点数，需转换为时间
def frequence_min(data):
    return np.argmin(np.abs(np.fft.fft(data)))
def frequence_max(data):
    return np.argmax(np.abs(np.fft.fft(data)[1:]))+1
#平均频段功率
def frequence_avg(data):
    return np.average(np.abs(np.fft.fft(data)))
def SpectralEntropy(data,subframesize):
    data = data.reshape((-1, subframesize))
    data = np.average(data ** 2, -1)
    return - np.sum(data * np.log10(data))

def SpectralFlux(data,subframesize):
    data = np.abs(np.fft.fft(data))
    data = data.reshape((-1, subframesize))
    data = data/(np.sum(data,-1)+1E-5)

    return np.log10(np.sum(np.ediff1d(data)**2))

def EnergyEntropy(data,subframesize):
    data = data.reshape((-1,subframesize))
    E_list = np.sum(data**2,-1)/(np.sum(data,-1) ** 2)
    E_list = E_list/np.sum(E_list)
    return - np.sum(E_list*np.log10(E_list))

def WAMP(data,threshold):
    return np.sum(-np.ediff1d(data)>threshold)


def RMS(data,subframesize):
    data = np.sum(data.reshape((-1,subframesize)),-1)
    return np.sqrt(np.average(data**2))

# 斜率变化率
def SSC(data):
    return len(np.unique((np.ediff1d(data)*100).astype(int)))

# def ZCR(data):
#     return np.average(np.ediff1d(np.sign(data)))
#
# def MAV(data):
#     return np.average(np.abs(data))
#
# def MMAX(data):
#     return np.max(np.abs(data))
#
# def MMIN(data):
#     return np.min(np.abs(data))
#
# def Mvariance(data):
#     return np.average((data - np.mean(np.abs(data))) ** 2)





def findpeaks(data, spacing=1, limit=None):
    """

    Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param ndarray data: data
    :param float spacing: minimum spacing to the next peak (should be 1 or more)
    :param float limit: peaks should have value greater or equal
    :return array: detected peaks indexes array
    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1
    x[-spacing:] = data[-1] - 1
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def searchPeaks(data):
    datalen = len(data)

    differentiated_emg_measurements = np.ediff1d(data)
    peak_candidate = np.zeros(datalen)
    peak_candidate[:] = False
    for i in range(len(differentiated_emg_measurements)-1):
        if differentiated_emg_measurements[i] >= 0 and differentiated_emg_measurements[i + 1] < 0:
            peak_candidate[i+1] = True

    peak_ind = np.argwhere(peak_candidate)
    peak_ind = peak_ind.reshape(peak_ind.size)
    peak_ind = np.unique(peak_ind)

    bottom_candidate = np.zeros(datalen)
    bottom_candidate[:] = False
    for i in range(len(differentiated_emg_measurements) - 1):
        if differentiated_emg_measurements[i] <= 0 and differentiated_emg_measurements[i + 1] > 0:
            bottom_candidate[i + 1] = True


    bottom_ind = np.argwhere(bottom_candidate)
    bottom_ind = bottom_ind.reshape(bottom_ind.size)
    bottom_ind = np.unique(bottom_ind)


    # plt.plot(data)
    # plt.plot(peak_ind, data[peak_ind], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    # plt.plot(bottom_ind, data[bottom_ind], 'rx', marker='x', color='#00008b', label='bottom', markersize=12)
    # plt.show()

    if len(bottom_ind) == 0 or len(peak_ind) == 0:
        # print('++++++++++++++++++++')
        return peak_ind, bottom_ind






    # l = -1
    # while len(peak_ind) != l:
    #     l = len(peak_ind)
    #     limit = np.sum(data[peak_ind])/l
    #     peak_ind = peak_ind[data[peak_ind] > limit*0.75]
    #
    # l = -1
    # while len(bottom_ind) != l:
    #     l = len(bottom_ind)
    #     b_limit = np.sum(data[bottom_ind])/l
    #     bottom_ind = bottom_ind[data[bottom_ind] < b_limit*1.4]

    # plt.plot(data)
    # plt.plot(peak_ind, data[peak_ind], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    # plt.plot(bottom_ind, data[bottom_ind], 'rx', marker='x', color='#00008b', label='bottom', markersize=12)
    # plt.show()

    if len(bottom_ind) == 0 or len(peak_ind) == 0:
        #print('++++++++++++++++++++')
        return peak_ind, bottom_ind

    i = 0
    j = 0
    pre_is_peak = None
    new_peak_ind = []
    new_bottom_ind = []
    while i<len(peak_ind) or j<len(bottom_ind):
        if i==len(peak_ind) or j==len(bottom_ind):
            if i == len(peak_ind):
                new_bottom_ind.append(bottom_ind[j])
            if j == len(bottom_ind):
                new_peak_ind.append(peak_ind[i])
            break
        if peak_ind[i] < bottom_ind[j]:
            if pre_is_peak is None or not pre_is_peak:
                new_peak_ind.append(peak_ind[i])
            pre_is_peak = True
            i = i + 1
        elif peak_ind[i] > bottom_ind[j]:
            if pre_is_peak is None or pre_is_peak:
                new_bottom_ind.append(bottom_ind[j])
            pre_is_peak = False
            j = j + 1
    peak_ind = new_peak_ind.copy()
    bottom_ind = new_bottom_ind.copy()
    if len(bottom_ind)==0 or len(peak_ind)==0:
        #print('++++++++++++++++++++')
        return peak_ind,bottom_ind

    # plt.plot(data)
    # plt.plot(peak_ind, data[peak_ind], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    # plt.plot(bottom_ind, data[bottom_ind], 'rx', marker='x', color='#00008b', label='bottom', markersize=12)
    # plt.show()

    # print(bottom_ind)
    # print(peak_ind)
    if peak_ind[0]<bottom_ind[0]:
        for i in range(len(peak_ind)):
            if i == 0:
                new_peak_ind[i] = np.argmax(data[:bottom_ind[i]]) + 0
            else:
                if i+1<len(bottom_ind):
                    new_peak_ind[i] = np.argmax(data[bottom_ind[i-1]:bottom_ind[i]])+bottom_ind[i-1]
                else:
                    new_peak_ind[i] = np.argmax(data[bottom_ind[i - 1]:peak_ind[i]+10]) + bottom_ind[i - 1]
        for i in range(len(bottom_ind)-1):
            if i+1 <len(peak_ind):
                new_bottom_ind[i] = np.argmin(data[peak_ind[i]:peak_ind[i+1]])+peak_ind[i]
            else:
                new_bottom_ind[i] = np.argmin(data[peak_ind[i]:bottom_ind[i]+10]) + peak_ind[i]


    if bottom_ind[0]<peak_ind[0]:
        for i in range(len(peak_ind)):
            if i == 0:
                new_bottom_ind[i] = np.argmin(data[:peak_ind[i]]) + 0
            else:
                if i + 1 < len(peak_ind):
                    new_bottom_ind[i] = np.argmin(data[peak_ind[i-1]:peak_ind[i]])+peak_ind[i-1]
                else:
                    new_bottom_ind[i] = np.argmin(data[peak_ind[i - 1]:bottom_ind[i]+10]) + peak_ind[i - 1]
        for i in range(len(peak_ind)-1):
            if i + 1 < len(bottom_ind):
                new_peak_ind[i] = np.argmax(data[bottom_ind[i]:bottom_ind[i+1]])+bottom_ind[i]
            else:
                new_peak_ind[i] = np.argmax(data[bottom_ind[i]:peak_ind[i]+10]) + bottom_ind[i]

    # plt.plot(data)
    # plt.plot(new_peak_ind, data[new_peak_ind], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    # plt.plot(new_bottom_ind, data[new_bottom_ind], 'rx', marker='x', color='#00008b', label='bottom', markersize=12)
    # plt.show()

    # index = np.unique(np.concatenate((new_peak_ind, new_bottom_ind)))
    # peaks = data[index]
    # index_ternal = np.ediff1d(index)
    # peaks_ternal = np.abs(np.ediff1d(peaks))
    # condidate = np.logical_and(index_ternal > np.mean(index_ternal) * 0.8, peaks_ternal > np.mean(peaks_ternal) * 0.8)
    # print(condidate)
    # new_index = []
    # for i, need in enumerate(condidate):
    #     if need:
    #         new_index.append(index[i])
    #         new_index.append(index[i + 1])
    # new_index = np.unique(new_index)
    # edif = np.ediff1d(data[new_index])
    # i = 0
    # index = [new_index[0]]
    # while i < len(edif) - 1:
    #     if edif[i + 1] * edif[i] < 0:
    #         index.append(new_index[i + 1])
    #     i += 1
    # index.append(new_index[-1])
    return peak_ind,bottom_ind



def dataProcess(data, winsize, type):
    """
    get the period feature of 'data' in certain 'winsize' with the 'type' of mean, max or min.
    :param ndarray data: data
    :param int winsize: data split winsize
    :param str type: one of mean, max or min
    :return ndarray: mean, max, or min feature array
    """
    dataLength = len(data)
    if type =='mean':
        stMeans = []
        for i in range(int(dataLength/WINSIZE)):
            stMean = np.mean(data[i*WINSIZE:(i+1)*WINSIZE-1])
            stMeans.append(stMean)
        return stMeans
    elif type == 'max':
        stMaxs = []
        for i in range(int(dataLength/WINSIZE)):
            stMax = np.max(data[i*WINSIZE:(i+1)*WINSIZE-1])
            stMaxs.append(stMax)
        return stMaxs
    elif type == 'min':
        stMins = []
        for i in range(int(dataLength / WINSIZE)):
            stMin = np.min(data[i * WINSIZE:(i + 1) * WINSIZE - 1])
            stMins.append(stMin)
        return stMin
    else:
        return data


def dataPlot(dataFilePath):
    data = np.load(dataFilePath)
    print(data)

    # plt.plot(data)


# dataRecordPeriod: Data recording duration such as 20 min
def data2csv(dataFilePath, dataRecordPeriod):
    path = str(dataFilePath)
    print(path)
    fatigueTimePath = path.replace('data', 'time')
    # print(fatigueTimePath)
    fatigueTime = np.load(fatigueTimePath, allow_pickle=True)
    # print(fatigueTime)
    diffTime = calculateDiffTime(path, fatigueTime)

    data = np.load(dataFilePath)
    dataLength = len(data)
    dataPerSecond = dataLength / dataRecordPeriod
    fatigueIndex = int(diffTime * dataPerSecond)
    # print(fatigueIndex)

    # 0-不疲劳，1-疲劳
    label = 0
    dataCsv = open(path.replace('npy', 'csv'), 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(dataCsv)
    csv_writer.writerow(['data','label'])
    for i in range(len(data)):
        if i >= fatigueIndex and label == 0:
            label = 1
        csv_writer.writerow([int(data[i]), label])
    dataCsv.close()


# dataRecordTime : data file path
def calculateDiffTime(dataRecordTime,fatigueTime):
    MatchPattern = 'D:/zwp/EMGData/(.*)_COM[56]_data.npy'
    RecordTime = re.match(MatchPattern, dataRecordTime).group(1)
    # print(RecordTime)
    dataStructTime = time.strptime(RecordTime, '%Y_%m_%d_%H-%M-%S')
    recordTime = time.mktime(dataStructTime)
    diffTime = recordTime - fatigueTime
    return diffTime


def readCsv(path):
    data = pd.read_csv(path)
    print(data)
    return data


def gettopdata(y):
    data = []
    for i in range(1, len(y) - 1):
        if y[i] > 0 and y[i - 1] < y[i] < y[i + 1]:
            data.append(i)
    return data


def findall(a):
    result = []
    data = []
    for i in range(1, len(a)):
        data.append(a[i] - a[i - 1])
    print(data)
    for pos in range(5, len(data) // 3):
        for i in range(len(data) - pos):
            if abs(data[i] - data[i + pos]) < abs(data[i]) * 0.05:
                # if data[i] == data[i+pos]:
                getp = True
                p = [a[i]]
                r = 1

                while r < (len(data) - i) // pos:

                    for j in range(pos):
                        if abs(data[i + j] - data[i + pos * r + j]) > abs(data[i + j]) * 0.05:
                            # if data[i+j] == data[i+pos*r+j]:
                            getp = False
                            break

                    if not getp:
                        break
                    else:
                        p.append(a[i + pos * r])
                        r += 1

                if len(p) > 4:
                    print(pos)
                    print(p)
                    result.append(p)

    print(result)
    return result


def bandpass_filter( data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def convert():
    dataDirectory = 'D:/zwp/EMGData/'
    fileList = os.listdir(dataDirectory)
    for file in fileList:
        if file.find('data') != -1:
            # print(file)
            data2csv(dataDirectory + file, RecordPeriod)

    print('convert succeed')





if __name__ == "__main__":



    # 提取区间内最大值
    # plt.subplot(1, 2, 1)
    # dataPlot('D:/EMGData/2021_04_06_15-06-26_com3_data.npy')
    # plt.subplot(1, 2, 2)
    # processData = dataProcess(np.load('D:/EMGData/2021_04_06_15-06-26_com3_data.npy'),'max')
    # plt.plot(processData)
    # plt.show()
    #
    #
    # 显示疲劳和非疲劳数据
    data = readCsv('D:/zwp/EMGData/2021_04_22_21-48-39_com3_data.csv')
    #
    #
    # emg = data['data'][:1200]
    # #emg = (emg - np.min(emg)) / (np.max(emg) - np.min(emg))
    # before = emg
    # from scipy import signal
    #
    # b, a = signal.butter(8, 0.1, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    # emg = signal.filtfilt(b, a, emg)
    # index = searchPeaks(emg)
    # # emg = np.convolve(emg, np.ones((5))) / 5
    # # emg = (emg - np.min(emg)) / (np.max(emg) - np.min(emg))
    #
    # plt.plot(before, linestyle = '-.', color = 'r')
    # plt.plot(emg, 'b')
    # #plt.plot(index, emg[index], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    # plt.show()
    #
    #
    #
    #
    # index = searchPeaks(emg)
    # plt.plot(index, emg[index], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    # plt.plot(emg,'b')
    # plt.plot(data_nf, 'b')
    # plt.plot(data_f, 'r')
    # plt.show()
    #
    # 训练模型

    x_train = np.array(data['data'])
    # 归一化
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    y_train = np.array(data['label'])
    y_train = y_train.reshape((-1, 1))
    x_data = []
    y_data = []
    i = 0
    while i*5+60 < len(x_train):
        x_data.append(x_train[i*5:i*5+60])
        y_data.append([1 if (np.count_nonzero(y_train[i*5:i*5+60])/30)>=1 else 0])
        i += 1
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(x_data)
    print(x_data.shape)
    print(y_data)
    print(y_data.shape)
    #
    #
    model = get_model()
    # model = keras.models.load_model("my_model")
    #
    # adam = opt.Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.CrossEntropyLoss()
    rand = np.arange(y_data.shape[0])
    print(rand)
    np.random.shuffle(rand)
    x_data = x_data[rand]
    y_data = y_data[rand]

    history = model.fit(
        x_data,
        y_data,
        batch_size=128,
        epochs=3200,
        validation_split=0.1
    )

    # model.save("my_model")
    # model = loadModel()
    # print(model.predict(x_data))

    # plt.plot(history.epoch,history.history.get('loss'))
    # plt.show()
    # plt.plot(history.epoch, history.history.get('acc'))
    # plt.show()
    #
    # plt.plot(data_f, 'r')
    # # data_nf = data_nf[data_nf>0]
    # # data_f = data_f[data_f>0]
    # len_nf = len(data_nf)
    # len_f = len(data_f)
    # # v=integrate.trapz(data_nf,range(len_nf))
    # # print(v/len_nf)
    # # v = integrate.trapz(data_f, range(len_f))
    # # print(v/len_f)
    # plt.show()
    #
    # 傅里叶变换
    # fft_data = np.fft.fft(data['data'])
    # abs_y = np.abs(fft_data)  # 取复数的绝对值，即复数的模(双边频谱)
    # angle_y = np.angle(fft_data)  # 取复数的角度
    # normalization_y = abs_y / n  # 归一化处理（双边频谱）
    # fre = 0
    # normalization_half_y = normalization_y[range(int(n / 2))]  # 由于对称性，只取一半区间（单边频谱）
    # maxFre = max(normalization_half_y[1:])
    # for i in range(len(normalization_half_y)):
    #     if normalization_half_y[i] == maxFre:
    #         fre = SAMPLEFZ*(i-2)/(len(normalization_half_y)-1)
    # t = 1/fre
    # print(t)
    # plt.plot(normalization_y[1:])
    # plt.show()
    #
    # 提取峰值
    # data = readCsv('D:/EMGData/2021_04_06_15-06-26_com3_data.csv')
    # top = gettopdata(data['data'])
    # findall(top)
    #
    # for i in indexs:
    #     plt.scatter(i - indexs[0] + 1, data['data'][i + 1])
    #
    # plt.plot(data['data'][indexs[0]:indexs[-1]])
    #
    # plt.show()
    #
    # data = readCsv('D:/EMGData/2021_04_08_16-04-35_com3_data.csv')
    # peak,bottom = searchPeaks(data['data'])
    # index = np.unique(np.concatenate((peak,bottom)))
    # plt.plot(data['data'])
    # plt.plot(index,data['data'][index],linestyle='-.',color='r')
    # plt.show()
    # peakInterval = np.ediff1d(peakIndex)
    # valleyInterval = np.ediff1d(bottomIndex)
    # plt.subplot(2,1,1)
    # plt.plot(peakInterval)
    # print(peakInterval)
    # plt.subplot(2,1,2)
    # plt.plot(valleyInterval)
    # plt.show()
    # print(peakIndex)
    # print(bottomIndex)
    #
    # print(peakIndex)
    # differentiated_emg_measurements = np.ediff1d(data['data'])
    # # Squaring - intensifies values received in derivative.
    # squared_ecg_measurements = differentiated_ecg_measurements ** 2
    # plt.subplot(2,1,1)
    # plt.plot(data['data'])
    # plt.subplot(2,1,2)
    # plt.plot(peakLoc)
    # plt.show()






