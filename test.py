# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
import time
from keras.models import load_model
import pandas as pd  

warnings.filterwarnings("ignore")

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

if __name__=='__main__':
    start = time.time()
    
    # load the dataset
    data_xls = pd.read_excel(u'流动性数据.xlsx', 'Sheet1', index_col=None)  
    data_xls.to_csv(u'预测.csv', encoding='utf-8')
    
    dataframe = read_csv(u'预测.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe[0:311].values
    
    # 将整型变为float
    dataset = dataset.astype('float32')

    # 当激活函数为 sigmoid 或者 tanh 时，要把数据正则话，此时 LSTM 比较敏感
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # split into train and test sets
    train_size = 249
    test_size = 32  #5月有21天
#    train, test = dataset[0:train_size,:], dataset[len(dataset)-test_size:len(dataset),:]
    train, test = dataset[0:train_size,:], dataset[train_size:train_size + test_size,:]

    # use this function to prepare the train and test datasets for modeling
    look_back = 10  #look_back 就是预测下一步所需要的 time steps
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
#    model = Sequential()  #线性叠加模型
#    model.add(LSTM(10, input_shape=(1, look_back)))  #隐藏层有 10 个神经元
#    model.add(Dense(1))
#    model.add(Activation("sigmoid"))
#    model.compile(loss='mean_squared_error', optimizer='adam')
#    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
#    
#    # 保存模型
#    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#    del model  # deletes the existing model

    model = load_model('my_model.h5')

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 计算误差之前要先把预测数据转换成同一单位
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])   

    # 计算 mean squared error 均方误差
#    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#    print('Train Score: %.2f RMSE' % (trainScore))
#    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#    print('Test Score: %.2f RMSE' % (testScore))
    
    print len(testY[0])
    print len(testPredict[:,0])
    print testY.shape
    print testPredict.shape

#    sum=0
#    for i in range(len(testY[0])):
#        sum+=abs((testY[0,i]-testPredict[i,0])/testY[0,i])
#    sum/=len(testY[0])
#    print ("error:",sum)
        
        

    # shift train predictions for plotting
#    trainPredictPlot = numpy.empty_like(dataset)
#    trainPredictPlot[:, :] = numpy.nan
#    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
#    testPredictPlot = numpy.empty_like(dataset)
#    testPredictPlot[:, :] = numpy.nan
##    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#    testPredictPlot[len(trainPredict)+(look_back)+1:len(dataset)-1, :] = testPredict
    
    end = time.time()
    print ("Total Time:",end - start)

    # plot baseline and predictions
    x=[1]*61
    for i in range(1,61):
        x[i]=i 
        
#    x=[1]*len(testPredict[:,0])
#    for i in range(1,len(testPredict[:,0])):
#        x[i]=i 

#    plt.plot(scaler.inverse_transform(test[look_back+1:]),"x-",label="True Data",color="red",linewidth=2)
    plt.plot(scaler.inverse_transform(dataset[train_size:train_size+21]),"x-",label="True Data",color="red",linewidth=2)  
#    plt.plot(testY[0],"x-",label="True Data",color="red",linewidth=2)
    plt.plot(testPredict,"+-",label="Test Prediction",color="blue",linewidth=2) 
    
#    plt.plot(scaler.inverse_transform(dataset), label='True Data')
#    plt.plot(trainPredictPlot, label='Train Prediction')
#    plt.plot(testPredictPlot, label='Test Prediction')

    plt.xlabel("Date")  
    plt.ylabel("Amount")  
    plt.title("Prediction") 
    plt.xlim(0, 25)
    plt.legend()
    plt.show()
    plt.savefig('result.png')