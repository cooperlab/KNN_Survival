%
% CORRELATION-BASED FEATURE SELECTION (filter method)
%

clear; close all; clc;

% add relevant paths
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
WhichModel = 'IDHwt';
%WhichModel = 'IDHmutCodel';
%WhichModel = 'IDHmutNonCodel';
%WhichModel = 'BRCA_Unprocessed';
%WhichModel = 'BRCA_Basic';
%WhichModel = 'BRCA_Reduced';

%% Read in data

if strcmp(WhichModel, 'Reduced') == 1
    load 'ReducedModel.mat';
    Features = ReducedModel.Features;
    Survival = ReducedModel.Survival +3;
    Censored = ReducedModel.Censored;
    Symbols = ReducedModel.Symbols;
    SymbolTypes = ReducedModel.SymbolTypes;
    
elseif strcmp(WhichModel, 'GBM') == 1
    load 'GBM_Preprocessed.mat';
    Features = GBM_Preprocessed.Features;
    Survival = GBM_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = GBM_Preprocessed.Censored;
    Symbols = GBM_Preprocessed.Symbols;
    SymbolTypes = GBM_Preprocessed.SymbolTypes;

elseif strcmp(WhichModel, 'LGG') == 1
    load 'LGG_Preprocessed.mat';
    Features = LGG_Preprocessed.Features;
    Survival = LGG_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = LGG_Preprocessed.Censored;
    Symbols = LGG_Preprocessed.Symbols;
    SymbolTypes = LGG_Preprocessed.SymbolTypes;

elseif strcmp(WhichModel, 'IDHwt') == 1
    load 'IDHwt_Preprocessed.mat';
    Features = IDHwt_Preprocessed.Features;
    Survival = IDHwt_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = IDHwt_Preprocessed.Censored;
    Symbols = IDHwt_Preprocessed.Symbols;
    SymbolTypes = IDHwt_Preprocessed.SymbolTypes;

elseif strcmp(WhichModel, 'IDHmutCodel') == 1
    load 'IDHmutCodel_Preprocessed.mat';
    Features = IDHmutCodel_Preprocessed.Features;
    Survival = IDHmutCodel_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = IDHmutCodel_Preprocessed.Censored;
    Symbols = IDHmutCodel_Preprocessed.Symbols;
    SymbolTypes = IDHmutCodel_Preprocessed.SymbolTypes;
    
elseif strcmp(WhichModel, 'IDHmutNonCodel') == 1
    load 'IDHmutNonCodel_Preprocessed.mat';
    Features = IDHmutNonCodel_Preprocessed.Features;
    Survival = IDHmutNonCodel_Preprocessed.Survival +3; %add 3 to ignore negative survival
    Censored = IDHmutNonCodel_Preprocessed.Censored;
    Symbols = IDHmutNonCodel_Preprocessed.Symbols;
    SymbolTypes = IDHmutNonCodel_Preprocessed.SymbolTypes;    
    
elseif strcmp(WhichModel, 'BRCA_Reduced') == 1
    load 'BRCA_ReducedModel.mat';
    Features = ReducedModel.Features;
    Survival = ReducedModel.Survival +3;
    Censored = ReducedModel.Censored;
    Symbols = ReducedModel.Symbols;
    SymbolTypes = ReducedModel.SymbolTypes;
    
end

% remove NAN survival or censorship values
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

[p,N] = size(Features);

%% Determine parameters and thresholds

Feat_Thresh = 300; %number of features to keep

K_min = 5; 
K_max = 20;

Filters = 'None';
%Filters = 'Both'; %choose this if performing gradient descent on sigma

Descent = 'None'; %fast
%Descent = 'Beta'; %slow, especially with more features
%Descent = 'sigma'; %slow, especially with more features
            
            Beta_init = ones(length(Feat_Thresh(:,1)),1); %initial beta (shrinking factor for features)
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

 
%% Get correlation with survival in uncensored cases in training+validation sets
%% 

Features_forCorr = [X_prototype, X_valid];
Survival_forCorr = [Survival_prototype, Survival_valid];
Censored_forCorr = [Censored_prototype, Censored_valid];

% Delete censored cases for correlating features
Survival_forCorr(Censored_forCorr == 1) = [];

j = 0;
for i = 1:length(Censored_forCorr)    
    if Censored_forCorr(1,i) == 1
        Features_forCorr(:,i-j) = [];
        j = j+1;
    end    
end

% Get spearman correlation
RHO = corr(Features_forCorr',Survival_forCorr','type','Spearman');

% Sort by absolute correlation
RHO(:,2) = 1:length(RHO); %feature index
RHO = abs(RHO);
RHO = sortrows(RHO,1);

% Remove NAN values and features
Delete1 = isnan(RHO(:,1));
j = 0;
for i = 1:length(Delete1)    
    if Delete1(i,1) == 1
        RHO(i-j,:) = [];
        j = j+1;
    end    
end

%% Pick top features

RHO = RHO(length(RHO) - Feat_Thresh+1 :end, :);


% Keep relevant features

Features_new_prototype = zeros(length(RHO),length(X_prototype(1,:)));
Features_new_valid = zeros(length(RHO),length(X_valid(1,:)));
Features_new_test = zeros(length(RHO),length(X_test(1,:)));

step = 1;
for i = 1:length(RHO)
    Features_new_prototype(step,:) = X_prototype(RHO(i,2),:);   
    Features_new_valid(step,:) = X_valid(RHO(i,2),:);
    Features_new_test(step,:) = X_test(RHO(i,2),:);
    step = step +1;
end

%% Update training, validation and testing feature sets

X_prototype = Features_new_prototype;
X_valid = Features_new_valid;
X_test = Features_new_test;


    %% Determine optimal model parameters using validation set
    
    % Determine optimal K
    K_star = 0;
    Accuracy_star = 0;
    
    for K = K_min:2:K_max
        
        clc
        trial
        K 
                      
        Y_valid_hat = KNN_Survival3(X_valid,X_prototype,Alive_prototype,K,Beta_init,Filters,sigma_init);
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
    if strcmp(Descent,'Beta') ==1
        Beta_star = KNN_Survival_Decend2b(X_valid,X_prototype,Alive_prototype,Alive_valid,K_star,Beta_init,Filters,Gamma_Beta,Pert_Beta,Conv_Thresh_Beta,sigma_init);
    elseif strcmp(Descent,'sigma') ==1    
        sigma_star = KNN_Survival_Decend2a(X_valid,X_prototype,Alive_prototype,Alive_valid,K_star,Beta_init,Filters,Gamma_sigma,Pert_sigma,Conv_Thresh_sigma,sigma_init);      
    end
    
    %% Determining testing accuracy (c-index) using testing set
    
    Alive_test_hat = KNN_Survival3(X_test,X_prototype,Alive_prototype,K_star,Beta_star,Filters,sigma_star);
    Alive_test_hat = sum(Alive_test_hat);
    C(trial,1) = cIndex2(Alive_test_hat,Survival_test,Censored_test);
    % mean squared error
    MSE(trial,1) = mean((Alive_test_hat(Censored_test==0) - Survival_test(Censored_test==0)) .^ 2);

end
