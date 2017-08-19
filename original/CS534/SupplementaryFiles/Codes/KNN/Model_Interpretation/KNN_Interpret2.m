%
% MODEL INTERPRETATION - GBMLGG (ENSEBMLE wrapper method)
%

clear; close all; clc;

% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/BRCA/')


%% Choose which model to use

%WhichModel = 'Reduced';
%WhichModel = 'GBM';
WhichModel = 'LGG';
%WhichModel = 'IDHwt';
%WhichModel = 'IDHmutCodel';
%WhichModel = 'IDHmutNonCodel';
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

% Make decisions on type of patient/feature set to use
Remove_mRNA = 0; %if == 0, set higher threshold for feature removal (0.99)


%% Remove mRNA features

if Remove_mRNA == 1
Feat_lim = 399; %limit of non-mRNA features (399 for reduced models)
Features(Feat_lim:end,:) = [];
Symbols(Feat_lim:end,:) = [];
SymbolTypes(Feat_lim:end,:) = [];
end

%% 
[p,N] = size(Features);

%% Determine parameters and thresholds - KNN

K_min = 15; %15 
K_max = 60; %70

Filters = 'None';
sigma_init = 7;
Lambda = 1; %the less the higher penality on lack of common dimensions
% Parameters for gradient descent on beta
Gamma_Beta = 15; %learning rate
Pert_Beta = 5; %this controls how much to perturb beta to get a feeling for gradient
Conv_Thresh_Beta = 0.0001; %convergence threshold 
Gamma_sigma = 10; %learning rate
Pert_sigma = 0.1; %this controls how much to sigma beta to get a feeling for gradient
Conv_Thresh_sigma = 0.0005; %convergence threshold for sigma
Descent = 'None'; 

%%  Set other parameters
trial_No = 10; % no of times to shuffle

Feat_Thresh = 0.99; % threshold of feature inclusion in each subset (0.9 means 10% of features enter model)
                    % 0.9 is no mRNA and 0.99 id with mRNA
Ensemble_No = 500; % number of random feature sets to generate

% Number of features in final model
% same size as ensembles
Model_size = ceil((1 - Feat_Thresh) .* length(Features(:,1)));
% OR fixed pre-set number
% Model_size = 40;

%% Generate feature ensembles

Feat_Include = rand(length(Features(:,1)),Ensemble_No);
Feat_Include = Feat_Include > Feat_Thresh;


%% Begin analysis

Feat_rank = zeros(length(Features(:,1)),trial_No+1);

for trial = 1:trial_No

    %% Shuffle samples

    Idx_New = randperm(N,N);
    
    Features = Features(:,Idx_New);
    Survival = Survival(:,Idx_New);
    Censored = Censored(:,Idx_New);
    

    %% Assign samples to PROTOTYPE set and validation set:
    %  The reason we call it "prototype set" rather than training set is 
    %  because there is no training involved. Simply, the patients in the 
    %  validation/testing set are matched to similar ones in the prototype
    %  ("database") set.
    
    K_cv = 2;
    Folds = ceil([1:N] / (N/K_cv));

    X_prototype = Features(:, Folds == 1);
    X_valid = Features(:, Folds == 2);
    
    Survival_prototype = Survival(:, Folds == 1);
    Survival_valid = Survival(:, Folds == 2);

    Censored_prototype = Censored(:, Folds == 1);
    Censored_valid = Censored(:, Folds == 2);

    % Convert outcome from survival to alive/dead status using time indicator
    t_min = min(Survival)-1;
    t_max = max(Survival);
    time = [t_min:1:t_max]';
    Alive_prototype = TimeIndicator(Survival_prototype,Censored_prototype,t_min,t_max);
    Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);

    
    %% Find the optimal K-value for the "typical" feature ensemble

    % Keeping only the subset of features within ensemble
    Keep_feat = Feat_Include(:,1);
    X_prototype_typical = X_prototype(Keep_feat == 1, :);
    X_valid_typical = X_valid(Keep_feat == 1, :);

    Beta1 = ones(length(X_valid_typical(:,1)),1);

    %
    % Find optimal K value
    %

    K_star = 0;
    Accuracy_star = 0;

    for K = K_min:2:K_max

        clc
        trial
        K 

        Y_valid_hat = KNN_Survival3(X_valid_typical,X_prototype_typical,Alive_prototype,K,Beta1,Filters,sigma_init);
        Y_valid_hat = sum(Y_valid_hat);
        Accuracy = cIndex2(Y_valid_hat,Survival_valid,Censored_valid);

        if Accuracy > Accuracy_star
            K_star = K;
            Accuracy_star = Accuracy;
        end

    end
    
        
    %% Looping through ensembles and calculating validation accuracy
    
    Ensemble_Accuracy = zeros(1,Ensemble_No);
    
    for Ensemble = 1:Ensemble_No
    
    clc
    trial
    Ensemble
        
    % Keeping only the subset of features within ensemble
    Keep_feat = Feat_Include(:,Ensemble);
    X_prototype_current = X_prototype(Keep_feat == 1, :);
    X_valid_current = X_valid(Keep_feat == 1, :);
    
    Beta1 = ones(length(X_valid_current(:,1)),1);
    
    % Calculating accuracy for current ensemble
    Y_valid_hat = KNN_Survival3(X_valid_current,X_prototype_current,Alive_prototype,K_star,Beta1,Filters,sigma_init);
    Y_valid_hat = sum(Y_valid_hat);
    Ensemble_Accuracy(1,Ensemble) = cIndex2(Y_valid_hat,Survival_valid,Censored_valid);
    
    end
    
    %% Weight each feature by its contribution to accurate ensembles
    
    % find average accuracy associated with each feature
    [Ensemble_Accuracy,~] = meshgrid(Ensemble_Accuracy,1:length(Features(:,1)));
    Ensemble_Accuracy = Ensemble_Accuracy .* Feat_Include;
    
    Feat_Accuracy_sum = sum(Ensemble_Accuracy, 2);
    Feat_Accuracy_count = sum(Feat_Include, 2);
    
    Feat_Accuracy_mean = Feat_Accuracy_sum ./ Feat_Accuracy_count;
    
    % Ignore features that were not included in any ensembles
    Feat_Accuracy_mean(isnan(Feat_Accuracy_mean)==1) = 0;
    
    % Add feature index
    Feat_Accuracy_mean(:,2) = [1:length(Features(:,1))]';
    
    % sort features by accuracy
    Feat_Accuracy_mean = sortrows(Feat_Accuracy_mean, 1);
    
    % save final feature ranks for current trial
    Feat_Accuracy_mean(:,3) = [length(Features(:,1)) : -1 : 1]';
    f_rank = zeros(length(Features(:,1)),1);
    f_rank(Feat_Accuracy_mean(:,2),1) = Feat_Accuracy_mean(:,3);
    
    Feat_rank(:,trial) = f_rank;
    
    

end

% find median rank over all trials
Feat_rank_median(:,1) = [1:length(Features(:,1))]'; %feature index
Feat_rank_median(:,2) = median(Feat_rank, 2);

% sort by median rank
Feat_rank_median = sortrows(Feat_rank_median, 2);

% Getting names of important features (IF NEEDED)
for i = 1:length(Features(:,1))

    Important_features{i,1} = Symbols{Feat_rank_median(i,1),1};
    Important_features{i,2} = SymbolTypes{Feat_rank_median(i,1),1};
    Important_features{i,3} = Feat_rank_median(i,2);
end



%     Feat_Best = Feat_Accuracy_mean( end - Model_size +1 : end, 2);