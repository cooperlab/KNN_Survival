clear; close all; clc;

% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/BRCA/')

%turn off warnings
warning('off','all')
 
%% Choose which model to use

%WhichModel = 'Unprocessed';
WhichModel = 'Basic';
%WhichModel = 'Reduced';
%WhichModel = 'GBM';
%WhichModel = 'LGG';
%WhichModel = 'IDHwt';
%WhichModel = 'IDHmutCodel';
%WhichModel = 'IDHmutNonCodel';
%WhichModel = 'BRCA_Unprocessed';
%WhichModel = 'BRCA_Basic';
%WhichModel = 'BRCA_Reduced';

%% Read in data

if strcmp(WhichModel, 'Basic') == 1
    load 'BasicModel.mat';
    Features = BasicModel.Features;
    Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
    Censored = BasicModel.Censored;

elseif strcmp(WhichModel, 'Reduced') == 1
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
    
elseif strcmp(WhichModel, 'BRCA_Basic') == 1
    load 'BRCA_BasicModel.mat';
    Features = BasicModel.Features;
    Survival = BasicModel.Survival +9;
    Censored = BasicModel.Censored;

elseif strcmp(WhichModel, 'BRCA_Reduced') == 1
    load 'BRCA_ReducedModel.mat';
    Features = ReducedModel.Features(1:100,:);
    Survival = ReducedModel.Survival +9;
    Censored = ReducedModel.Censored;
    Symbols = ReducedModel.Symbols;
    SymbolTypes = ReducedModel.SymbolTypes;
    
end

%Clean up the nan value in survival & censored
Keep = ~isnan(Survival) & ~isnan(Censored) & (sum(isnan(Features), 1) == 0);
Features = Features(:, Keep);
Survival = Survival(Keep);
Censored = Censored(Keep);
[p,N] = size(Features);
 
 
trial_No = 10; % no of times to shuffle
 
%%
 
C = zeros(trial_No,1);
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
    K_cv = 2;
    Folds = ceil([1:N] / (N/K_cv));

    
    X_train = Features(:, Folds == 1);
    X_test = Features(:, Folds == 2);
 
    Survival_train = Survival(:, Folds == 1);
    Survival_test = Survival(:, Folds == 2);
 
    Censored_train = Censored(:, Folds == 1);
    Censored_test = Censored(:, Folds == 2);

        clc
        trial

        BetaBasic = coxphfit(X_train',Survival_train','Censoring',Censored_train);
        
        %%
       
    C(trial,1) = cIndex(BetaBasic, X_test',Survival_test, Censored_test);
    
end

% turn warnings back on
warning('on','all')