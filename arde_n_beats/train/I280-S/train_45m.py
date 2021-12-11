# -*- coding: utf-8 -*-
import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import backend as K
from processing.processing import processing
import numpy as np
import random
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import load_model
from evaluate.evaluate import evaluation
import math
import pathlib
from datetime import datetime
os.chdir(os.path.dirname(__file__))


class config:
    '''
    configure parameters and paths
    '''
    TrainValRatio = [0.8, 0.2]
    date_start = '2018-04-01'
    date_split = '2019-01-18'
    datapath = '../../data/I280-S/D_45m.csv'
    ParaSavePath = '../../hypara/I280-S/parameter_d3.csv'
    stopmarginloss = 0.003
    stopmarginloss_decay = 0.8
    threhold = 0.06

    # batchsize,timestep,block_per_stack,hidden,layer,dim,lr,lr_f,patience
    bounds_all = [(10,512),(1,50),(1,10),(1,500),(1,10),(1,20),(0.0001,0.1),(0,1),(1,10)]
    F_c = 0.7
    EarlyStopStep = 3
    maxiter = 10
    F_list = [(0.8, 1), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2)]
    popsize = 20
    seed = 1109


# set seeds
np.random.seed(config.seed)
random.seed(config.seed)
os.environ['PYTHONHASHSEED']=str(config.seed)


def dataset(timestep):
    '''
    extract data for training
    :param timestep: timesteps for the extracted data
    :return: inp train; oup train; inp validation; oup validation
    '''
    flow_data = pd.read_csv(config.datapath)
    x_train,y_train = processing(flow_data, timestep, config.date_start, config.date_split, train=True).main()

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

    return x_train_,y_train_,x_val_,y_val_


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, Global_fir_epoch_loss, config, step):
        '''
        :param Global_fir_epoch_loss: a dynamic parameter of the global minimum loss of the first training epoch
        :param config: config class
        :param step: generation of ARDE algorithm
        '''
        super(tf.keras.callbacks.Callback, self).__init__()
        self.Global_fir_epoch_loss = Global_fir_epoch_loss
        self.config = config
        self.step = step

    def on_epoch_end(self, epoch, logs={}):
        '''
        function to conduct early stop depending on the first training epoch loss
        :param epoch: training epoch
        :return: None
        '''
        val_loss = logs.get('val_loss')
        threshold = config.stopmarginloss * (config.stopmarginloss_decay ** (self.step + 1))
        if val_loss>self.Global_fir_epoch_loss+threshold or math.isnan(val_loss):
            print('early stop!!!')
            self.model.stop_training = True


def Train(x, D, Global_fir_epoch_loss, round_no, f, num, step):
    '''
    train an N-BEATS model
    :param x: hyperparameter list
    :param D: data set symbol
    :param Global_fir_epoch_loss: a dynamic parameter of the global minimum loss of the first training epoch
    :param round_no: round number
    :param f: a global text file to record the training process
    :param num: candidate index for each generation
    :param step: generation
    :return: validation MAPE; first epoch loss
    '''
    tf.random.set_seed(config.seed)
    from nbeats.model import NBeatsNet

    batchsize = int(round(x[0]))
    timestep = int(round(x[1]))
    block_per_stack = int(round(x[2]))
    hidden = int(round(x[3]))
    layer = int(round(x[4]))
    dim = int(round(x[5]))
    lr = round(x[6], 4)
    lr_f = round(x[7], 4)
    patience = int(round(x[8]))

    thetas_dim = tuple([dim]*layer)

    type_ = NBeatsNet.GENERIC_BLOCK
    stack_types = tuple([type_]*layer)

    LastIdx = [idx for idx in range(50)]
    ValdIdx = LastIdx[-timestep:]

    x_train_loc = x_train_glo[:, ValdIdx, :]
    y_train_loc = y_train_glo.copy()

    x_val_loc = x_val_glo[:, ValdIdx, :]
    y_val_loc = y_val_glo.copy()

    min_flow = x_train_loc.min()
    max_flow = x_train_loc.max()

    # data normilization
    x_train_loc = (x_train_loc - min_flow) / (max_flow - min_flow)
    y_train_loc = (y_train_loc - min_flow) / (max_flow - min_flow)
    x_val_loc= (x_val_loc - min_flow) / (max_flow - min_flow)
    y_val_loc = (y_val_loc - min_flow) / (max_flow - min_flow)

    # call N-BETAS model
    model = NBeatsNet(backcast_length=timestep, forecast_length=1,
                      stack_types=stack_types, nb_blocks_per_stack=block_per_stack,
                      thetas_dim=thetas_dim, share_weights_in_stack=True, hidden_layer_units=hidden)

    opt = optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)

    # shuffle training data
    np.random.seed(1239)
    x_train_loc = np.random.permutation(x_train_loc)
    np.random.seed(1239)
    y_train_loc = np.random.permutation(y_train_loc)

    # save model
    string = ""
    for i in range(len(x)):
        if i in [0,1,2,3,4,5,8]:
            item = int(round(x[i]))
        else:
            item = round(x[i], 4)
        string = string + '_' + str(item)

    SaveModlPath = '../../save/I280-S/D_' + str(D) + '/round_'+str(round_no)+'/'
    SaveModlFile = SaveModlPath + 'model' + string + '.h5'
    pathlib.Path(SaveModlPath).mkdir(parents=True, exist_ok=True)

    # define callbacks
    cus_callback = CustomCallback(Global_fir_epoch_loss, config, step)
    mcp_save = callbacks.ModelCheckpoint(SaveModlFile, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=lr_f,patience=patience,min_lr=0.0001,mode='min')

    # fit N-BEATS model
    if num == 0 and step == 0:
        history = model.fit(x=x_train_loc, y=y_train_loc, epochs=150,
                batch_size=batchsize, validation_data=(x_val_loc, y_val_loc),
                callbacks=[mcp_save, reduce_lr_loss],verbose=0)
    else:
        history = model.fit(x=x_train_loc, y=y_train_loc, epochs=150,
                batch_size=batchsize, validation_data=(x_val_loc, y_val_loc),
                callbacks=[mcp_save, reduce_lr_loss, cus_callback], verbose=0)

    # get the loss of the first training epoch
    fir_epoch_loss = history.history["val_loss"][0]

    # load model
    try:
        model = load_model(SaveModlFile)
    except:
        pass

    # prediction and evaluation
    predictions_val = model.predict(x_val_loc, batch_size=batchsize)
    predictions_val = predictions_val*(max_flow-min_flow)+min_flow
    _, _, mape_val = evaluation(predictions_val, y_val_glo)

    if math.isnan(mape_val):
        mape_val = 1000

    K.clear_session()
    del model

    printtxt = "MAPE: %.4f: %s,%s,%s,%s,%s,%s,%s,%s,%s" % (
        mape_val,batchsize,timestep,block_per_stack,hidden,layer,dim,lr,lr_f,patience)
    print(printtxt)
    os.write(f, str.encode(printtxt + '\n'))

    return mape_val,fir_epoch_loss


def RunARDE(D,round_no):
    '''
    run ARDE algorithm
    :param D: data set symbol
    :param round_no: round number
    :return: None
    '''
    start = datetime.now()
    from de.differential_evolution import DEAlgorithm

    DEAlgorithmClass = DEAlgorithm(config)
    InitialArray = DEAlgorithmClass.initialization()

    GlobalMinValLi = []
    BestMape = []
    GlobalMinVal = 1000

    SaveTextPath = '../../text/I280-S/D_' + str(D) + '/round_'+str(round_no)+'/'
    pathlib.Path(SaveTextPath).mkdir(parents=True, exist_ok=True)
    f = os.open(SaveTextPath + "prcoess.txt", os.O_RDWR | os.O_CREAT)

    Global_fir_epoch_loss = 0
    Global_step_mape = 1000
    for step in range(config.maxiter):
        printtxt = 'Data: %s, Round: %s, Step %s' % (D,round_no,step)
        print(printtxt)
        os.write(f, str.encode(printtxt + "\n"))

        if step == 0:
            args = InitialArray
            CombineArray = InitialArray
        else:
            F_l = config.F_list[step][0]
            F_u = config.F_list[step][1]
            MutatedArray = DEAlgorithmClass.mutation(InitialArray, F_l, F_u)
            CrossArray = DEAlgorithmClass.crossover(InitialArray, MutatedArray)

            args = CrossArray
            CombineArray = InitialArray + CrossArray

        mapelist = []
        m = 0
        newinitialarray = []
        for num in range(len(args)):
            item = args[num]
            res,fir_epoch_loss = Train(item, D, Global_fir_epoch_loss, round_no, f, num, step)
            mapelist.append(res)

            if num==0 and step==0:
                Global_fir_epoch_loss = fir_epoch_loss
                Global_step_mape = res
            else:
                if res<Global_step_mape:
                    Global_fir_epoch_loss = fir_epoch_loss
                    Global_step_mape = res

            if step==0 and res<config.threhold:
                m = m + 1
                newinitialarray.append(item)
                BestMape.append(res)
                if m>=config.popsize:
                    break

        if step == 0:
            mapelist = mapelist
        else:
            mapelist = BestMape + mapelist

        StepMinVal = min(mapelist)

        if StepMinVal < GlobalMinVal:
            LIdx = mapelist.index(StepMinVal)
            StepOptimalPara = CombineArray[LIdx]
            GlobalOptimalPara = StepOptimalPara
            GlobalMinVal = StepMinVal

        printtxt = "Step: %s, min MAPE: %s, hyperparameter: %s" % (step, GlobalMinVal, GlobalOptimalPara)
        print(printtxt)
        os.write(f, str.encode(printtxt + '\n'))

        if step==0:
            InitialArray = newinitialarray
        else:
            SelectArray, BestMape = DEAlgorithmClass.selection(mapelist, InitialArray, CrossArray)
            InitialArray = SelectArray

        # early stopping applied
        GlobalMinValLi.append(GlobalMinVal)
        if step + 1 >= config.EarlyStopStep:
            if GlobalMinValLi[-1] == GlobalMinValLi[-2] == GlobalMinValLi[-3]:
                break

    duration = (datetime.now() - start).total_seconds()/3600
    printtxt = "computational time: %s h" %(duration)
    print(printtxt)

    # update best para table
    Df_Para = pd.read_csv(config.ParaSavePath)
    Df_step = pd.DataFrame({'round': [round_no], 'best_para': [GlobalOptimalPara], 'time': [duration], 'best_fitness': [GlobalMinVal]})
    Df_Para = pd.concat([Df_Para, Df_step], axis=0)
    Df_Para.to_csv(config.ParaSavePath, index=False)

    return None


if __name__ == "__main__":
    global x_train_glo, y_train_glo, x_val_glo, y_val_glo
    x_train_glo, y_train_glo, x_val_glo, y_val_glo = dataset(50)

    RunARDE(3, 0)