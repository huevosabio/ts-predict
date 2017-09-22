# imports
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import math
import keras
import time
from itertools import product
from scipy.stats  import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, LSTM, Dropout, RepeatVector, Layer
from keras.regularizers import l2
from keras        import backend as K
from datetime     import datetime, timedelta
from keras.optimizers      import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics       import mean_squared_error
from matplotlib.pylab      import rcParams

# the transform class, helps transforming data
class Transform:
    def __init__(self, feature_range=(-1,1), frequency='day', log=True):
        self.feature_range = feature_range
        self.log = log
        if frequency not in ['day', 'week']: 
            raise Exception('Frequency must be either day or week')
        self.frequency = frequency
        if frequency == 'day':
            self.freq = 288
        elif frequency == 'week':
            self.freq = 288 * 7
        
    def fit(self, dataset):
        '''
        Assumes dataset is a pandas dataframe
        '''
        # log transform
        if self.log:
            dlog = np.log(dataset + 1)
        self.seasonals = dlog.copy()
        self.columns = dlog.columns
        for c in dlog.columns:
            res = seasonal_decompose(dlog[c], freq=self.freq)
            self.seasonals[c] = res.seasonal
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        self.scaler.fit(dlog.values - self.seasonals.values)
        if self.frequency == 'day':
            self.seasonals['time'] = self.seasonals.index.time
            self.seasonals = self.seasonals.groupby('time').mean()
        elif self.frequency == 'week':
            self.seasonals['time'] = self.seasonals.index.hour * 60 + \
            self.seasonals.index.minute + self.seasonals.index.weekday * 24 * 60
            self.seasonals = self.seasonals.groupby('time').mean()
        return
        
    def transform(self, dataset):
        '''
        Assumes dataset is a pandas dataframe
        '''
        # box transform
        if self.log:
            dlog = np.log(dataset + 1)
        # get the corresponding seasonals
        if self.frequency == 'day':
            index = dlog.index.time
        elif self.frequency == 'week':
            index = dlog.index.hour * 60 + dlog.index.minute + dlog.index.weekday * 24 * 60
        # subtract seasonality
        dlog[self.columns] = dlog.values - self.seasonals.loc[index,dlog.columns].values
        # minmax scaler
        return self.scaler.transform(dlog.values)
    
    def inverse_transform(self, values, datetime_index):
        # inverse mimmax scaler
        vals = self.scaler.inverse_transform(values)
        # get the seasonality
        dlog = pd.DataFrame(vals, index=datetime_index, columns=self.columns)
        if self.frequency == 'day':
            index = dlog.index.time
        elif self.frequency == 'week':
            index = dlog.index.hour * 60 + dlog.index.minute + dlog.index.weekday * 24 * 60
        # add the seasonality
        dlog[self.columns] = dlog.values + self.seasonals.loc[index,dlog.columns].values
        # inverse box transform
        if self.log:
            return np.exp(dlog) - 1
        else:
            return dlog

# lstm model
def create_model(ncols, look_back, look_forward, drate=0.25, rdrate=0.3, weight_decay=1e-7):

    model = Sequential()
    #model.add(Dropout(
    #    rate,
    #    input_shape=(look_back, stations+2),
    #))
    model.add(LSTM(
        ncols*2,
        input_shape=(look_back, ncols),
        return_sequences=True,
        recurrent_dropout=rdrate,
        kernel_regularizer = l2(weight_decay),
        bias_regularizer = l2(weight_decay),
        recurrent_regularizer = l2(weight_decay),
    ))  # Encoder 1
    model.add(Dropout(
        drate,
    ))
    model.add(LSTM(
        ncols / 2,
        return_sequences=False,
        recurrent_dropout=rdrate,
        kernel_regularizer = l2(weight_decay),
        bias_regularizer = l2(weight_decay),
        recurrent_regularizer = l2(weight_decay),
    ))  # Encoder 2
    model.add(Dropout(
        drate,
    ))
    model.add(RepeatVector(look_forward))
    model.add(Dropout(
        drate,
    ))
    model.add(LSTM(
        ncols*2,
        return_sequences=True,
        recurrent_dropout=rdrate,
        kernel_regularizer = l2(weight_decay),
        bias_regularizer = l2(weight_decay),
        recurrent_regularizer = l2(weight_decay),
    ))  # Decoder
    model.add(Dropout(
        drate,
    ))
    model.add(LSTM(
        ncols*2,
        return_sequences=True,
        recurrent_dropout=rdrate,
        kernel_regularizer = l2(weight_decay),
        bias_regularizer = l2(weight_decay),
        recurrent_regularizer = l2(weight_decay),
    ))  # Decoder
    model.add(Dropout(
        drate,
    ))
    model.add(LSTM(
        ncols*2,
        return_sequences=True,
        recurrent_dropout=rdrate,
        kernel_regularizer = l2(weight_decay),
        bias_regularizer = l2(weight_decay),
        recurrent_regularizer = l2(weight_decay),
    ))  # Decoder
    model.add(Dropout(
        drate,
    )) 
    model.add(TimeDistributed(Dense(
        ncols,
        #activation="relu",
        kernel_regularizer = l2(weight_decay),
        bias_regularizer = l2(weight_decay),
    )))
    start = time.time()
    model.compile(loss="mse", optimizer="adam")
    print "Compilation Time : ", time.time() - start
    model.summary()
    print "Inputs: {}".format(model.input_shape)
    print "Outputs: {}".format(model.output_shape)
    return model

# TB callback
tbCallBack = lambda name: keras.callbacks.TensorBoard(
    log_dir='ignored_assets/tf-log/' + str(name),
    histogram_freq=0,write_graph=True, write_images=True
)

def create_dataset(dataset, look_back=1, look_forward=2, not_in_Y=0):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        # Makes sure that the array isn't the last 2 or 3 or whatever bits. It has to be the full 24
        if len(dataset[i + look_back:i+look_back+look_forward, 0]) == look_forward:
            dataX.append(a)
            dataY.append(dataset[i + look_back:i+look_back+look_forward,:])
    return np.array(dataX), np.array(dataY)

def predict_with_uncertainty(model, dataset, ncols, n_iter=100):
    # This function activates drop-out when doing predictions
    f = K.function([model.layers[0].input, K.learning_phase()],
                      [model.layers[-1].output])
    all_pred = np.zeros((n_iter,) + (look_forward,ncols,) )
    for i in range(n_iter):
        all_pred[i] = np.array( f([dataset.reshape( (1,) + dataset.shape), 1]) ).reshape(look_forward, ncols)
    avg_pred = all_pred.mean(axis=0)
    std = all_pred.std(axis=0)
    return all_pred, avg_pred, std

# given a model and data from the original dataset, it uses predict_with_uncertainty to predict with dropout
def run_predictions(model, dataset, transformer, prediction_tindex):

    # scale the dataset2
    data = transformer.transform(dataset)

    # feeds the model and data into a function that will activate the dropout
    all_pred, avg_pred, std = predict_with_uncertainty(model, data, dataset.shape[-1])
    # All the predictions
    #dummy_cols = np.zeros((all_pred.shape[1], 2))
    for i in range(all_pred.shape[0]):
        all_pred[i,:,:] = transformer.inverse_transform(all_pred[i,:,:], prediction_tindex)
    # The Standard Deviation At Each T Of Prediction
    #std = all_pred.std(axis=0)
    std_u = transformer.inverse_transform(avg_pred + std, prediction_tindex).values
    std_b = transformer.inverse_transform(avg_pred - std, prediction_tindex).values
    std_u2 = transformer.inverse_transform(avg_pred + 2*std, prediction_tindex).values
    std_b2 = transformer.inverse_transform(avg_pred - 2*std, prediction_tindex).values
    std = transformer.inverse_transform(std, prediction_tindex).values
    # The Average Prediction
    #avg_pred = all_pred.mean(axis=0)
    avg_pred = transformer.inverse_transform(avg_pred, prediction_tindex).values
    #print all_pred.mean(axis=0), avg_pred

    return all_pred, avg_pred, std, std_u, std_b, std_u2, std_b2

# takes a look_back, and look_forward and 
def train_model(look_back, look_forward, df, train_size):
    # create transform (transformer should only look at training data)
    transformer = Transform(frequency='day')
    transformer.fit(df.iloc[:train_size,:])
    dataset = transformer.transform(df)
    # split into train and test sets
    test_size = len(dataset) - train_size
    test, train = dataset[0:test_size,:], dataset[test_size:len(dataset)-look_forward,:]
    # reshape into X=[t, t-1, t-2,..., t-look_back] and Y=[t+1, t+2,... t+look_forward]
    trainX, trainY = create_dataset(train, look_back, look_forward)
    testX, testY = create_dataset(test, look_back, look_forward)
    # create and fit the LSTM network
    model = create_model(trainX.shape[-1], look_back, look_forward)
    # trains the model
    model.fit(
        trainX, trainY, 
        epochs=30, batch_size=50, 
        verbose=3, validation_data=(testX, testY), 
        callbacks=[tbCallBack('dayseas_to_'+str(look_back)+'b_'+str(look_forward)+'f')]
        )
    # evaluates the model and prints the result
    print 'Train Error', model.evaluate(trainX, trainY, verbose=2)
    print 'Test Error', model.evaluate(testX, testY, verbose=2)
    model.transformer = transformer
    return model

def batch_predict(model, datasets, transformer, prediction_indices):
    '''
        model is a trained keras model
        datasets is a list of valid dataframes
        transformer is the data transformer
        prediction_indices is a list of prediction indices
    '''
    batch_size = len(datasets)
    look_forward = len(prediction_indices[0])
    ncols = datasets[0].shape[-1]
    normalized_input = np.zeros((batch_size,) + datasets[0].shape)
    for i,d in enumerate(datasets):
        normalized_input[i,:,:] = transformer.transform(d)
    # predict
    output = model.predict(normalized_input, 50)
    dfs = []
    for i in range(output.shape[0]):
        dfs.append(transformer.inverse_transform(output[i,:,:], prediction_indices[i]))
    return dfs

# takes a dataframe and makes the required predictions
def predict_for_date(prediction_day, dataframe, model, look_back, look_forward):
    start = prediction_day - timedelta(minutes=5)
    # one prediction per time step
    npred = 289
    stepsize = timedelta(minutes=5)
    tindex = [start + i*stepsize for i in range(npred)]
    # make the batches
    dfs = []
    pats = []
    tstamps = []
    for i,present in enumerate(tindex):
        # the point where the model will predict
        past = present - (look_back-1) * timedelta(minutes=5)
        ptindex = [present + t*timedelta(minutes=5) for t in range(1,look_forward+1)]
        dfs.append(dataframe.loc[past:present,:])
        pats.append([present]*len(ptindex))
        tstamps.append(ptindex)
    
    predictions = pd.concat(batch_predict(model, dfs, model.transformer, pats))
    predictions['predicted_at'] = np.concatenate(pats)
    predictions['timestamp'] = np.concatenate(tstamps)
    return predictions

def load_df():
    # 1 distribution by source & time
    # load data
    dataframe = pd.read_csv('ignored_assets/paxout_table.csv', engine='python', nrows=288*21)
    # create map of core stations
    dataframe['time_bucket'] = pd.to_datetime(dataframe['time_bucket'])
    dataframe = dataframe.set_index('time_bucket')
    return  dataframe

def main():
    print 'Hi, there. Loading data...'
    ## load data
    df = load_df()
    print 'Done!'
    ## train model
    # params
    train_size = 288*15
    horizons = [3, 6, 12, 24, 48]
    prediction_day = datetime(2016, 1, 21)
    print 'Begin running the predictions, this will take a looong time...'
    for look_back, look_forward in product(horizons, horizons):
        print 'Start with look_back:',look_back,' and look_forward',look_forward
        print 'model training'
        t1 = time.time()
        model = train_model(look_back, look_forward, df, train_size)
        t2 = time.time()
        print 'done, it took ', t2 - t1, 's'
        print 'running predictions'
        predictions = predict_for_date(prediction_day, df, model, look_back, look_forward)
        t3 = time.time()
        print 'done, it took ', t3 - t2, 's'
        print 'saving...'
        predictions.to_csv('ignored_assets/to_lstm_'+str(look_back)+'b_'+str(look_forward)+'f.csv', index=False)
        K.clear_session()
    print 'Done! Thanks for waiting'


print 'where are my print statements?'
if __name__=='__main__':
    print 'Starting...'
    main()
    
