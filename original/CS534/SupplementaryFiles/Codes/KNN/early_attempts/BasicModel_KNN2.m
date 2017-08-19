%
% Assess performance of basic model using training/validation/testing
% approach with shuffling - KNN using area under curve * tmax (to get
% ACTUAL survival times)
%
% NOTE: Codes used here are a combination of original codes and codes
% provided in the "PerformanceExample.m" script provided by Dr Lee Cooper
% in CS-534 class
%
% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')

% turn off warnings
% warning('off','all')

%% Using basic model as provided by Dr Lee (PerformanceExample.m)

% read in data and extract minimal features
load 'BasicModel.mat';

Features = BasicModel.Features;
Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
Censored = BasicModel.Censored;

[p,N] = size(Features);

%% Determine initial parameters

K_min = 2; 
K_max = 100;

Beta1 = ones(length(Features(:,1)),1); %shrinking factor for features

trial_No = 10; % no of times to shuffle

%%

C = zeros(trial_No,1);
MSE = zeros(trial_No,1);

for trial = 1:trial_No

    %% Shuffle samples

    Idx_New = randperm(N,N);
    Features_New = zeros(size(Features));
    Survival_New = zeros(size(Survival));
    Censored_New = zeros(size(Censored));
    for i = 1:N
    Features_New(:,i) = Features(:,Idx_New(1,i));
    Survival_New(:,i) = Survival(:,Idx_New(1,i));
    Censored_New(:,i) = Censored(:,Idx_New(1,i));
    end
    Features = Features_New;
    Survival = Survival_New;
    Censored = Censored_New;

    %% Assign samples to PROTOTYPE set, validation set (for model selection) ... 
    %  and testing set (for model assessment):
    %  The reason we call it "prototype set" rather than training set is 
    %  because there is no training involved. Simply, the patients in the 
    %  validation/testing set are matched to similar ones in the prototype
    %  ("database") set.
    
    K_cv = 3;
    Folds = ceil([1:N] / (N/K_cv));

    X_prototype = Features(:, Folds == 1);
    X_valid = Features(:, Folds == 2);
    X_test = Features(:, Folds == 3);

    Survival_prototype = Survival(:, Folds == 1);
    Survival_valid = Survival(:, Folds == 2);
    Survival_test = Survival(:, Folds == 3);

    Censored_prototype = Censored(:, Folds == 1);
    Censored_valid = Censored(:, Folds == 2);
    Censored_test = Censored(:, Folds == 3);
    
    %% Determine optimal model parameters using validation set
    
    K_star = 0;
    Accuracy_star = 0;
    
    for K = K_min:2:K_max
        
        clc
        trial
        K
        
        Y_valid_hat = KNN_Survival2(X_valid,X_prototype,Survival_prototype,Censored_prototype,K,Beta1);
        Accuracy = cIndex2(Y_valid_hat,Survival_valid,Censored_valid);
        
        if Accuracy > Accuracy_star
            K_star = K;
            Accuracy_star = Accuracy;
        end
        
    end
    
    %% Determining testing accuracy (c-index) using testing set
    
    Y_test_hat = KNN_Survival2(X_test,X_prototype,Survival_prototype,Censored_prototype,K_star,Beta1);
    C(trial,1) = cIndex2(Y_test_hat,Survival_test,Censored_test);
    % mean squared error
    MSE(trial,1) = mean((Y_test_hat(Censored_test==0) - Survival_test(Censored_test==0)) .^ 2);
    
end
    