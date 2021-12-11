# -*- coding: utf-8 -*-
import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from processing.processing import processing
import numpy as np
import random
from nbeats.model import NBeatsNet
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from evaluate.evaluate import evaluation
os.chdir(os.path.dirname(__file__))


class config:
    '''
    configure parameters and paths
    '''
    TrainValRatio = [0.8, 0.2]
    date_start = '2018-04-01'
    date_split = '2019-01-18'
    datapath = '../../data/M1-N/D_30m.csv'
    ParaPath = '../../hypara/M1-N/parameter_d2.csv'
    SavePath = '../../hypara/M1-N/test_result_d2.csv'
    seed = 2560


# set seeds
np.random.seed(config.seed)
random.seed(config.seed)
os.environ['PYTHONHASHSEED']=str(config.seed)


def dataset(timestep):
    '''
    extract data for training
    :param timestep: timesteps for the extracted data
    :return: inp train; oup train; inp validation; oup validation; inp test; oup test
    '''
    flow_data = pd.read_csv(config.datapath)
    x_train, y_train, x_test, y_test = processing(flow_data, timestep, config.date_start, config.date_split,train=False).main()

    # split to train, val sets
    num_val = int(len(x_train) * config.TrainValRatio[1])

    IdxTrainVal = [idx for idx in range(0, x_train.shape[0])]

    random.seed(191)
    IdxVal = random.sample(IdxTrainVal, num_val)
    IdxTrain = [idx for idx in IdxTrainVal if idx not in IdxVal]

    x_train_ = x_train[IdxTrain, :, :]
    y_train_ = y_train[IdxTrain, :]
    x_val_ = x_train[IdxVal, :, :]
    y_val_ = y_train[IdxVal, :]

    return x_train_, y_train_, x_val_, y_val_, x_test, y_test


def Test(x,D,round_no):
    '''
    for testing
    :param x: hyperparameter list
    :param D: data set symbol
    :param round_no: round number
    :return: MAPE, RMSE and MAPE on test data
    '''
    batchsize = int(round(x[0]))
    timestep = int(round(x[1]))
    block_per_stack = int(round(x[2]))
    hidden = int(round(x[3]))
    layer = int(round(x[4]))
    dim = int(round(x[5]))
    lr = round(x[6], 4)
    lr_f = round(x[7], 4)
    patience = int(round(x[8]))

    thetas_dim = tuple([dim] * layer)

    type_ = NBeatsNet.GENERIC_BLOCK
    stack_types = tuple([type_] * layer)

    LastIdx = [idx for idx in range(50)]
    ValdIdx = LastIdx[-timestep:]

    x_train_loc = x_train_glo[:, ValdIdx, :]
    x_test_loc = x_test_glo[:, ValdIdx, :]

    min_flow = x_train_loc.min()
    max_flow = x_train_loc.max()

    # data normilization
    x_test_loc = (x_test_loc - min_flow) / (max_flow - min_flow)

    model = NBeatsNet(backcast_length=timestep, forecast_length=1,
                      stack_types=stack_types, nb_blocks_per_stack=block_per_stack,
                      thetas_dim=thetas_dim, share_weights_in_stack=True, hidden_layer_units=hidden)

    opt = optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)

    string = ""
    for i in range(len(x)):
        if i in [0,1,2,3,4,5,8]:
            item = int(round(x[i]))
        else:
            item = round(x[i], 4)
        string = string + '_' + str(item)

    SaveModlPath = '../../save/M1-N/D_' + str(D) + '/round_'+str(round_no)+'/'
    SaveModlFile = SaveModlPath + 'model' + string + '.h5'

    # load model
    model = load_model(SaveModlFile)

    predictions_test = model.predict(x_test_loc, batch_size=batchsize)
    predictions_test = predictions_test * (max_flow - min_flow) + min_flow
    mae_test, rmse_test, mape_test = evaluation(predictions_test, y_test_glo)

    predictions_test = np.squeeze(predictions_test, axis=2)
    y_test_save = np.squeeze(y_test_glo, axis=2)
    np.savetxt('../../result/M1-N/prediction_d2.csv', predictions_test, delimiter=',')
    np.savetxt('../../result/M1-N/groundtruth_d2.csv', y_test_save, delimiter=',')

    return mae_test, rmse_test, mape_test


if __name__ == "__main__":
    global x_train_glo, y_train_glo, x_val_glo, y_val_glo, x_test_glo, y_test_glo
    x_train_glo, y_train_glo, x_val_glo, y_val_glo, x_test_glo, y_test_glo = dataset(50)
    D = 2

    # read saved best hyper-parameters
    ParaTable = pd.read_csv(config.ParaPath, dtype='string')

    mae_li = []
    rmse_li = []
    mape_li = []
    time_li = []
    for index, row in ParaTable.iterrows():
        round_no = int(row['round'])
        time = float(row['time'])
        Para = row['best_para']
        Para_r = Para.replace("[", "").replace("]", "")
        Para_s = Para_r.split(", ")
        Para_f = [float(item) for item in Para_s]

        print('Round: %s' % (round_no))

        mae_test, rmse_test, mape_test = Test(Para_f, D, round_no)
        mae_li.append(mae_test)
        rmse_li.append(rmse_test)
        mape_li.append(mape_test)
        time_li.append(time)

    SaveDf = pd.DataFrame({'MAE': mae_li, 'RMSE': rmse_li, 'MAPE': mape_li, 'Time': time_li})
    SaveDf.to_csv(config.SavePath, index=False)
