# 记得装sklearn和netCDF4
# 上云的时候要检查使用的包是否都有！！！

print("Loading libs...")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import netCDF4 as nc
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten
#from tensorflow.keras.layers import Conv3D, MaxPool3D
from tensorflow.keras.layers import Dropout, Dense, LSTM, GlobalAveragePooling2D
#from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("All the libs are loaded.")

"""def Normalization(vector):
    norm = np.linalg.norm(vector)
    normalized_array = vector/norm
    return normalized_array"""

class ResnetBlock(Model):

    def __init__(self, filters, shape, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False, input_shape=shape)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False, input_shape=shape)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False, input_shape=shape)
            self.down_b1 = BatchNormalization()
        
        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, shape, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.shape = shape
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, kernel_size=3, strides=1, padding='same', use_bias=False, input_shape = shape[1:])
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, shape, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, shape, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        # self.p1 = GlobalAveragePooling2D()
        #self.f1 = Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        print("resnet:",inputs.shape)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        print("resnest:",x.shape)
        # y = self.p1(x)
        y = tf.reshape(x,(batch_size,-1,x.shape[-1]))   # 这句话改过了
        
        print("resnet:",y.shape)
        #y = self.f1(x)
        return y
print("ResNet18 have been successfully defined.")


nc_filepath1 = r"e:\(2020)adaptor.mars.internal-1660720321.0924718-25455-7-c67c5831-e169-47f6-aaca-646aebef8112.nc"
nc_filepath2 = r"e:\(2021)adaptor.mars.internal-1660451633.7985704-18099-7-5e23ce1c-db2c-4468-8b34-51f4bba35a0e.nc"
SPI3_filepath = r""
nc_file1 = nc.Dataset(nc_filepath1)
nc_file2 = nc.Dataset(nc_filepath2)
SPI3_file = open(SPI3_filepath, encoding='UTF-8')
# 不能对训练集与测试集进行打乱，因为我们的输入数据是时间序列

# 数据集四个括号的理解：
# >>> print(file.dimensions)
# >>> {'longitude': <class 'netCDF4._netCDF4.Dimension'>: name = 'longitude', size = 1440, 'latitude': <class 'netCDF4._netCDF4.Dimension'>: name = 'latitude', size = 721, 'level': <class 'netCDF4._netCDF4.Dimension'>: name = 'level', size = 4, 'time': <class 'netCDF4._netCDF4.Dimension'>: name = 'time', size = 5}

# 训练集60%（0-447），开发集20%（445-596），测试集20%（594-744）
# 或训练集80%（0-596），测试集20%（594-744）

# 高度从0-3: 200, 500, 750, 850
# 以下这种方式在输出的x.shape = (16, monthnum, 2884, 1440)
x = []
temp = []
keylist = list(nc_file1.variables.keys())

# monthnum_file1 = 32*12, 32 yrs & 12 months-per-yr
# monthnum_file2 = 31*12, 31 yrs & 12 months-per-yr
for key in keylist[4:]:
    loadingStart = time.perf_counter()
    for month in nc_file1.variables[key][:3]:
        # shape(month) = (4,721,1440), 3D
        # temp1 = np.reshape(month,(-1,1440)); shape(temp1) = (2884,1440), 2D
        temp.append(np.reshape(month,(-1,1440)))    # shape(temp) = (monthnum_file1,2884,1440), 3D
    for month in nc_file2.variables[key][:2]:
        # shape(month) = (4,721,1440)
        # temp1 = np.reshape(month,(-1,1440)); shape(temp1) = (2884,1440), 2D
        temp.append(np.reshape(month,(-1,1440)))    # shape(temp) = (monthnum_file1+monthnum_file2,2884,1440), 3D
    x.append(temp)                                  # shape(x) = (16,monthnum_file1+monthnum_file2,2884,1440), 4D
    temp = []
    loadingEnd = time.perf_counter()
    loadingTime = round(loadingEnd-loadingStart, 3)
    print("Successfully appended variable {} within {} secs.".format(key, loadingTime))

x = np.transpose(x, (1,3,2,0))                      # shape(x) = (monthnum_file1+monthnum_file2,16,2884,1440), 4D
train_test_sep = 3
x_train = np.array(x[:train_test_sep])
x_test = np.array(x[train_test_sep:])
print("*********The ERA5 dataset has been successfully loaded.*********")


str_lst = SPI3_file.readlines()
y = [float(x) for x in str_lst]
SPI3_file.close()

y_train = [0.3, 0.2, 0.6]
y_train = np.array(y_train)
y_test = [0.1, 0.5]
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 以下这种方式在输出的x.shape = (16, monthnum, 4, 721, 1440)，与Conv2D输入有差别，故舍
"""x = []
x_train = []
y_train = [1, 2, 3]
y_train = np.array(y_train)
x_test = []
y_test = [4, 5]
y_test = np.array(y_test)
train_test_sep = 3

keylist = list(nc_file1.variables.keys())
# 加载第一个nc文件中的数据
for key in keylist[4:]:      # 忽略前四个变量名是因为前四个变量名分别对应经度、纬度、高度、时间
    loadingStart = time.perf_counter()
    x.append(nc_file1.variables[key][:3])
    loadingEnd = time.perf_counter()
    loadingTime = round(loadingEnd-loadingStart, 3)
    #print("Successfully appended variable {} of 1959-1990 within {} secs.".format(key, loadingTime))
    #print(shape(nc_file.variables[key][0]))
    # 此处x的shape为(16, time, 4, 721, 1440)
print("*********Data of 1959-1990 has been successfully loaded*********")

# 加载第二个nc文件中的数据
i = 0
for key in keylist[4:]:      # 忽略前四个变量名是因为前四个变量名分别对应经度、纬度、高度、时间
    loadingStart = time.perf_counter()
    x[i] = np.append(x[i], nc_file2.variables[key][:2], axis=0)
    loadingEnd = time.perf_counter()
    loadingTime = round(loadingEnd-loadingStart, 3)
    #print("Successfully appended variable {} of 1991-2021 within {} secs.".format(key, loadingTime))
    i += 1
    #print(shape(nc_file.variables[key][0]))
del i
print("*********Data of 1991-2021 has been successfully loaded*********")
print("Shape of x is {}".format(np.shape(x)))
# 不用tf的转置而是用np的转置是因为数据量大时tf的转置容易爆显存，np转置使用cpu和内存就没有问题

# 为什么要对x_train和x_test升维？？？
#x_train = np.expand_dims(x_train, axis=0)
#x_test = np.expand_dims(x_test, axis=0)"""





batch_size = 1
CNN = ResNet18([2, 2, 2, 2], (batch_size, 1440, 2884, 16))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = tf.keras.models.Sequential([
        CNN,
        # tf.keras.layers.TimeDistributed(CNN),    # 将CNN网络打包在TD层中实现复用
        LSTM(80),
        Dropout(0.1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mean_squared_error') 

checkpoint_save_path = "./checkpoint/LRCN.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
print("pre_fit:",x_train.shape)
print("pre_fit:",x_test.shape)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test), 
                    validation_freq=1, callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
################## predict ######################
# 测试集输入模型进行预测
predicted_SPI3 = model.predict(x_test)
"""# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(test_set[60:])"""
# 画出真实数据和预测数据的对比曲线
plt.plot(y_test, color='red', label='SPI3')
plt.plot(predicted_SPI3, color='blue', label='Predicted SPI3')
plt.title('SPI3 Prediction')
plt.xlabel('Time')
plt.ylabel('SPI3')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_SPI3, y_test)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_SPI3, y_test))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_SPI3, y_test)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)

"""def LRCN(self):

    model = Sequential()
    
    model.add(TimeDistributed(Convolution2D(32, (7,7), strides=(2, 2),
        padding='same', activation='relu'), input_shape=self.input_shape))
    model.add(TimeDistributed(Convolution2D(32, (3,3),
        kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Convolution2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Convolution2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Convolution2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Convolution2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Convolution2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Convolution2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.7))
    model.add(LSTM(512, return_sequences=False, dropout=0.5))
    model.add(Dense(self.nb_classes, activation='softmax'))
            
    return model"""