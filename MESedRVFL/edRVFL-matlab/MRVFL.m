clc;clear all;
% function [Model,TrainAcc,TestAcc,TrainingTime,TestingTime]  = MRVFL(trainX,trainY,testX,testY,option)

% load('train_5403.mat');
load('DATA2.mat');
for i=1:length(sigXY)
    [labely(i),labelx(i)]=find(sigXY(i,:)==1);
end
trainX=sigXY(1:3406,1:15);
trainY=labelx(1,1:3406)';
testX=sigXY(3406:5403,1:15);
testY=labelx(1,3406:5403)';

option.L = 8;                 % Number of layers [int]
option.N = 2000;                % Number of neurons [int]
option.C = 0.1;                % Regularisation Parameter [float]
option.scale = 1;              % Scaling parameter [float]
option.Activation = "relu";    % Activation function ['relu','sigmoid']


% Requried for consistency
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

% Train RVFL
[Model,TrainAcc,TrainingTime] = MRVFLtrain(trainX,trainY,option);

% Using trained model, predict the testing data
[TestAcc,TestingTime] = MRVFLpredict(testX,testY,Model);

% end
%EOF