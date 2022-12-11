from MESRVFLvtrain import *

def MESRVFLv(trainX,trainY,data,option):

    # Train MESRVFLv
    [Model,TrainAcc,TrainingTime,Training_loss] = MESRVFLvtrain(trainX,trainY,data,option)


    return Model,TrainAcc,TrainingTime,Training_loss,
#EOF

