# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:44:40 2020

@author:
"""
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
import numpy as np
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 输入图像channel：1；输出channel：16；5x5卷积核
#         self.conv1 = nn.Conv1d(1, 16, 5)
#         self.conv2 = nn.Conv1d(16, 16, 5)
#
#         self.conv3 = nn.Conv1d(16, 32, 3)
#         self.conv4 = nn.Conv1d(32, 32, 3)
#
#         self.conv3 = nn.Conv1d(32, 256, 3)
#         self.conv4 = nn.Conv1d(256, 256, 3)
#
#
#         # an affine operation: y = W
#         # x + b
#         self.fc1 = nn.Linear(256, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)
#
#     def forward(self, x):
#         # 2x2 Max pooling
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = F.max_pool1d(x, 2)
#         # 如果是方阵,则可以只使用一个数字进行定义
#         x = F.dropout(x, p=0.1)
#
#         x = F.max_pool1d(self.conv4(self.conv3(x)), 2)
#         # 如果是方阵,则可以只使用一个数字进行定义
#         x = F.dropout(x, p=0.1)
#
#         x = nn.AdaptiveMaxPool1d(1)(self.conv6(self.conv5(x)))
#         # 如果是方阵,则可以只使用一个数字进行定义
#         x = F.dropout(x, p=0.2)
#
#         x = x.view(-1, self.num_flat_features(x))
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.sigmoid(self.fc3(x))
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # 除去批处理维度的其他所有维度
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


def get_model():
    config = tensorflow.compat.v1.ConfigProto(gpu_options=tensorflow.compat.v1.GPUOptions(allow_growth=True))
    sess = tensorflow.compat.v1.Session(config=config)
    nclass = 1
    inp = Input(shape=(60, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation="relu", padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation="relu", padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation="relu", padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation="relu", padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation="relu", name="dense_1")(img_1)
    dense_1 = Dense(64, activation="relu", name="dense_2")(dense_1)
    dense_1 = Dense(nclass,  activation="sigmoid", name="output")(dense_1)

    model = keras.models.Model(inputs=inp, outputs=dense_1)
    opt = keras.optimizers.Adam(1e-4)

    model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


# non_fatigues_x = np.load('non_fatigues_data.npy')
# non_fatigues_y = np.zeros((non_fatigues_x.shape[0],1))
#
#
#
# fatigues_x = np.load('fatigues_data.npy')
# fatigues_y = np.ones((fatigues_x.shape[0],1))
# x_train = np.concatenate((non_fatigues_x,fatigues_x))
# y_train = np.concatenate((non_fatigues_y,fatigues_y))
#
#
# x_train = np.load('jb_data_x.npy')
# y_train = np.load('jb_data_y.npy')
# y_train = y_train.reshape((-1,1))
#
# model = get_model()
# #model = keras.models.load_model("my_model")
#
# # adam = opt.Adam(model.parameters(), lr=0.001)
# # loss_fn = nn.CrossEntropyLoss()
# rand=np.arange(y_train.shape[0])
# np.random.shuffle(rand)
# x_train = x_train[rand]
# y_train = y_train[rand]
#
#
# # x_train = torch.DoubleTensor(x_train)
# # y_train = torch.DoubleTensor(y_train)
# # x_val = torch.tensor(x_val)
# # y_val = torch.tensor(y_val)
#
# #
# # for i in range(1000):
# #     output = model(x_train)
# #     loss = loss_fn(output, y_train)
# #     adam.zero_grad()
# #     loss.backward()
# #     adam.step()
#
#
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=50000,
#     epochs=3200,
#     validation_split=0.1
# )
#
# model.save("my_model")