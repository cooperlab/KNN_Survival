%
% Assess performance using training/validation/testing
% approach with shuffling - KNN_Survival4 using the Alive_train 
%
%
% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/BRCA/')


%% Choose which model to use

%WhichModel = 'Unprocessed';
%WhichModel = 'Basic';
%WhichModel = 'Reduced';
%WhichModel = 'GBM';
%WhichModel = 'LGG';
%WhichModel = 'IDHwt';
WhichModel = 'BRCA_Unprocessed';
%WhichModel = 'BRCA_Basic';
%WhichModel = 'BRCA_Reduced';

if strcmp(WhichModel, 'Unprocessed') == 1
    load 'GBMLGG.Data.mat';
    Survival = Survival +3; %add 3 to ignore negative survival
    
elseif strcmp(WhichModel, 'Basic') == 1
    load 'BasicModel.mat';
    Features = BasicModel.Features;
    Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
    Censored = BasicModel.Censored;

elseif strcmp(WhichModel, 'Reduced') == 1
    load 'ReducedModel.mat';
    Features = ReducedModel.Features;
    Survival = ReducedModel.Survival +3; %add 3 to ignore negative survival
    Censored = ReducedModel.Censored;

elseif strcmp(WhichModel, 'GBM') == 1
    load 'GBM_Preprocessed.mat';
    Features = GBM_Preprocessed.Features;
    Survival = GBM_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = GBM_Preprocessed.Censored;

elseif strcmp(WhichModel, 'LGG') == 1
    load 'LGG_Preprocessed.mat';
    Features = LGG_Preprocessed.Features;
    Survival = LGG_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = LGG_Preprocessed.Censored;

elseif strcmp(WhichModel, 'IDHwt') == 1
    load 'IDHwt_Preprocessed.mat';
    Features = IDHwt_Preprocessed.Features;
    Survival = IDHwt_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = IDHwt_Preprocessed.Censored;

elseif strcmp(WhichModel, 'IDHmutCodel') == 1
    load 'IDHmutCodel_Preprocessed.mat';
    Features = IDHmutCodel_Preprocessed.Features;
    Survival = IDHmutCodel_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = IDHmutCodel_Preprocessed.Censored;

elseif strcmp(WhichModel, 'BRCA_Unprocessed') == 1
    load 'BRCA.Data.mat';
    Survival = Survival +9; %add 3 to ignore negative survival

elseif strcmp(WhichModel, 'BRCA_Basic') == 1
    load 'BRCA_BasicModel.mat';
    Features = BasicModel.Features;
    Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
    Censored = BasicModel.Censored;

elseif strcmp(WhichModel, 'BRCA_Reduced') == 1
    load 'BRCA_ReducedModel.mat';
    Features = ReducedModel.Features;
    Survival = ReducedModel.Survival +3; %add 3 to ignore negative survival
    Censored = ReducedModel.Censored;

end

% remove NAN survival or censorship values
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

[p,N] = size(Features);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add NAN values at random to simulate missing data
% pNaN = 0.75; %proportion of NAN values
% 
% NaN_Idx = randperm(N*p,N*p); 
% NaN_Idx = NaN_Idx(1:pNaN * N*p);
% 
% Features(NaN_Idx) = nan;
% 
% % Using only a proportion of the features
% pFeat = 0.999; %proportion of features to delete
% 
% Del = zeros(1,p);
% Del_Idx = randperm(p,ceil(p * pFeat));
% Del(Del_Idx) = 1;
% Features(Del==1,:) = [];
%
% [p,N] = size(Features);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Standardize features of unprocessed model

if strcmp(WhichModel, 'Unprocessed') == 1 || strcmp(WhichModel, 'BRCA_Unprocessed') == 1
    
    % Remove features with zero variance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonNAN_sum = Features;
    nonNAN_sum(isnan(nonNAN_sum)==1) = 0;
    nonNAN_sum = sum(nonNAN_sum, 2);
    nonNAN_count = ~isnan(Features);
    nonNAN_count = sum(nonNAN_count, 2);
    nonNAN_mean = nonNAN_sum ./ nonNAN_count;
    [~,nonNAN_mean] = meshgrid(1:length(Features(1,:)), nonNAN_mean);

    Features_mean = nonNAN_mean;
    Features_var = (Features - nonNAN_mean) .^ 2;

    nonNAN_sum = Features_var;
    nonNAN_sum(isnan(nonNAN_sum)==1) = 0;
    nonNAN_sum = sum(nonNAN_sum, 2);
    nonNAN_count = ~isnan(Features_var);
    nonNAN_count = sum(nonNAN_count, 2);
    Features_var = nonNAN_sum ./ nonNAN_count;

    Features(Features_var == 0, :) = [];
    Symbols(Features_var == 0, :) = [];
    SymbolTypes(Features_var == 0, :) = [];

    Features_mean(Features_var == 0, :) = [];
    Features_var(Features_var == 0, :) = [];

    % Z- score standardization of features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,Features_var] = meshgrid(1:length(Features(1,:)), Features_var);
    Features = (Features - Features_mean) ./ (Features_var .^ 0.5);

end

%% Determine initial parameters

% basic model: 15 - 70
% BRCA basic: 20 - 40
% Reduced: 20 - 40
% BRCA reduced: 40 - 70
% Unprocessed: 30 - 50

K_min = 15; %15
K_max = 70; %70


Filters = 'None';
%Filters = 'Both'; %choose this if performing gradient descent on sigma

Descent = 'None'; %fast
%Descent = 'Beta'; %slow, especially with more features
%Descent = 'sigma'; %slow, especially with more features
    
        % Ignore if descent = None
        Beta_init = ones(length(Features(:,1)),1); %initial beta (shrinking factor for features)
        sigma_init = 7;

        Lambda = 1; %the less the higher penality on lack of common dimensions

        % Parameters for gradient descent on beta
        Gamma_Beta = 15; %learning rate
        Pert_Beta = 5; %this controls how much to perturb beta to get a feeling for gradient
        Conv_Thresh_Beta = 0.0001; %convergence threshold 

        Gamma_sigma = 10; %learning rate
        Pert_sigma = 0.1; %this controls how much to sigma beta to get a feeling for gradient
        Conv_Thresh_sigma = 0.0005; %convergence threshold for sigma


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

    % Convert outcome from survival to alive/dead status using time indicator
    t_min = min(Survival)-1;
    t_max = max(Survival);
    time = [t_min:1:t_max]';
    Alive_prototype = TimeIndicator(Survival_prototype,Censored_prototype,t_min,t_max);
    Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);
    
    %% Determine optimal model parameters using validation set
    
    % Determine optimal K
    K_star = 0;
    Accuracy_star = 0;
    
    for K = K_min:2:K_max
        
        clc
        trial
        K 
                      
        Y_valid_hat = KNN_Survival4(X_valid,X_prototype,Alive_prototype,K,Beta_init,Filters,sigma_init,Lambda);
        Y_valid_hat = sum(Y_valid_hat);
        Accuracy = cIndex2(Y_valid_hat,Survival_valid,Censored_valid);
        
        if Accuracy > Accuracy_star
            K_star = K;
            Accuracy_star = Accuracy;
        end
        
    end

    Beta_star = Beta_init;
    sigma_star = sigma_init;
    
    % Gradient descent on beta
%     if strcmp(Descent,'Beta') ==1
%         Beta_star = KNN_Survival_Decend2b(X_valid,X_prototype,Alive_prototype,Alive_valid,K_star,Beta_init,Filters,Gamma_Beta,Pert_Beta,Conv_Thresh_Beta,sigma_init);
%     elseif strcmp(Descent,'sigma') ==1    
%         sigma_star = KNN_Survival_Decend2a(X_valid,X_prototype,Alive_prototype,Alive_valid,K_star,Beta_init,Filters,Gamma_sigma,Pert_sigma,Conv_Thresh_sigma,sigma_init);      
%     end
    
    %% Determining testing accuracy (c-index) using testing set
    
    Alive_test_hat = KNN_Survival4(X_test,X_prototype,Alive_prototype,K_star,Beta_star,Filters,sigma_star,Lambda);
    Alive_test_hat = sum(Alive_test_hat);
    C(trial,1) = cIndex2(Alive_test_hat,Survival_test,Censored_test);
    % mean squared error
    MSE(trial,1) = mean((Alive_test_hat(Censored_test==0) - Survival_test(Censored_test==0)) .^ 2);
    
end
    