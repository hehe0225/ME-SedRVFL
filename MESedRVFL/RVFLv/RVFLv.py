from RVFLvtrain import *

def RVFLv(trainX,trainY,option):

    # Train SRVFLv
    [Model,TrainingTime,Training_loss] = RVFLvtrain(trainX,trainY,option)

    return Model,TrainingTime,Training_loss,

