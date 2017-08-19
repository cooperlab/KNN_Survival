%
% Feature reduction GEARED TOWARDS keeping an appropriate sample size for 
% different LGG subtypes !! - This emphasizes maximizing sample size
%

clear; close all; clc;

% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')

% turn off warnings
% warning('off','all')

%% Read in data and initial preprocessing and feature extraction

load 'GBMLGG.Data.mat';

% convert to better format
SymbolTypes = cellstr(SymbolTypes);
Symbols = cellstr(Symbols);
Samples = (cellstr(Samples))';

% Remove problematic features (lots of missing data affecting results)
Del_Idx = [1:length(Features(:,1))]';
del1 = Del_Idx(strcmp(Symbols, 'race-Is-white_Clinical'), 1);
del2 = Del_Idx(strcmp(Symbols, 'race-Is-black'), 1);
del3 = Del_Idx(strcmp(Symbols, 'race-Is-asian_Clinical'), 1);
del4 = Del_Idx(strcmp(Symbols, 'race-Is-american'), 1);
del5 = Del_Idx(strcmp(Symbols, 'radiation_therapy-Is-yes_Clinical'), 1);

Feat_Del = [del1,del2,del3,del4,del5];

Features(Feat_Del, :) = [];
SymbolTypes(Feat_Del, :) = [];
Symbols(Feat_Del, :) = [];

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

% Remove patients missing survival/censored data %%%%%%%%%%%%%%%%%%%%%%%%%%
Features(:,isnan(Survival)==1) = [];
Samples(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Samples(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

%
% IMPORTANT NOTE: %
%
% Some groups of patients are more emphasized by certain thresholds than
% others. The "ReducedModel.mat" has an under-representation of GBM. 
% Over here, different thresholds are used to maximize sample size in each 
% patient subpopulation.
%
%

% Remove patients missing information on GBM/LGG status
unknown1 = isnan(Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:));
unknown2 = isnan(Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:));
unknown3 = isnan(Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:));

isUnknown = (unknown1 + unknown2 + unknown3) > 0;

Features = Features(:, ~isUnknown);
Survival = Survival(:, ~isUnknown);
Censored = Censored(:, ~isUnknown);
Samples = Samples(:, ~isUnknown);


%% POPULATION 1: Isolate GBM patients


% Remove LGG patients
isGBM1 = Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:));
isGBM2 = Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:));
isGBM3 = Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:));

isGBM = (isGBM1 + isGBM2 + isGBM3) > 0;

Features_GBM = Features(:, isGBM);
Survival_GBM = Survival(:, isGBM);
Censored_GBM = Censored(:, isGBM);
Samples_GBM = Samples(:, isGBM);
Symbols_GBM = Symbols;
SymbolTypes_GBM = SymbolTypes;

% Remove GBM features (except treatment)
Del_Idx = [1:length(Features_GBM(:,1))]';
del1 = Del_Idx(strcmp(Symbols_GBM, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'), 1);
del2 = Del_Idx(strcmp(Symbols_GBM, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'), 1);

Feat_Del = [del1,del2];

Features_GBM(Feat_Del, :) = [];
Symbols_GBM(Feat_Del, :) = [];
SymbolTypes_GBM(Feat_Del, :) = [];

%                         % Visualize matrix to facilitate tuning threshold
%                         Features_visual = ~isnan(Features_GBM);
%                         f_visual(:,:,1) = zeros(size(Features_visual));
%                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                         f_visual(:,:,3) = zeros(size(Features_visual));
%                         figure(1)
%                         image(f_visual); 
%                         hold on;
%                         title('GBM: original');

% remove features missing more than a threshold of patients
Thesh_f = 0.5; % Best: 0.5
f = sum((isnan(Features_GBM)==1),2);

Features_GBM (f > (Thesh_f * length(Features_GBM(1,:))), :) = [];

Symbols_GBM(f > (Thesh_f * length(Features_GBM(1,:))), :) = [];
SymbolTypes_GBM (f > (Thesh_f * length(Features_GBM(1,:))), :) = [];



%                         % Visualize matrix to facilitate tuning threshold
%                         clear('Features_visual', 'f_visual')
%                         Features_visual = ~isnan(Features_GBM);
%                         f_visual(:,:,1) = zeros(size(Features_visual));
%                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                         f_visual(:,:,3) = zeros(size(Features_visual));
%                         figure(2)
%                         image(f_visual)
%                         hold on;
%                         title('GBM: feature removal');                        
                        
% remove patients missing more than a threshold of features
Thesh_p = 0.005; %Best: 0.005
p = sum((isnan(Features_GBM)==1),1);

Features_GBM (:, p > (Thesh_p * length(Features_GBM(:,1)))) = [];
Survival_GBM (:, p > (Thesh_p * length(Features_GBM(:,1)))) = [];
Censored_GBM (:, p > (Thesh_p * length(Features_GBM(:,1)))) = [];
Samples_GBM (:, p > (Thesh_p * length(Features_GBM(:,1)))) = [];


%                         % Visualize matrix to facilitate tuning threshold
%                         clear('Features_visual', 'f_visual')
%                         Features_visual = ~isnan(Features_GBM);
%                         f_visual(:,:,1) = zeros(size(Features_visual));
%                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                         f_visual(:,:,3) = zeros(size(Features_visual));
%                         figure(3)
%                         image(f_visual)
%                         hold on;
%                         title('GBM: patient removal');
                        
% Pack resultant matrix
GBM_Preprocessed.Features = Features_GBM;
GBM_Preprocessed.Symbols = Symbols_GBM;
GBM_Preprocessed.SymbolTypes = SymbolTypes_GBM; 
GBM_Preprocessed.Samples = Samples_GBM; 
GBM_Preprocessed.Survival = Survival_GBM;
GBM_Preprocessed.Censored = Censored_GBM;


%% POPULATION 2: Isolate LGG patients

% Remove GBM patients
isGBM1 = Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:));
isGBM2 = Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:));
isGBM3 = Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:));

isLGG = (isGBM1 + isGBM2 + isGBM3) == 0;

Features_LGG = Features(:, isLGG);
Survival_LGG = Survival(:, isLGG);
Censored_LGG = Censored(:, isLGG);
Samples_LGG = Samples(:, isLGG);
Symbols_LGG = Symbols;
SymbolTypes_LGG = SymbolTypes;

% Remove GBM features
Del_Idx = [1:length(Features_LGG(:,1))]';
del1 = Del_Idx(strcmp(Symbols_LGG, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'), 1);
del2 = Del_Idx(strcmp(Symbols_LGG, 'histological_type-Is-treated primary gbm_Clinical'), 1);
del3 = Del_Idx(strcmp(Symbols_LGG, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'), 1);

Feat_Del = [del1,del2,del3];

Features_LGG(Feat_Del, :) = [];
Symbols_LGG(Feat_Del, :) = [];
SymbolTypes_LGG(Feat_Del, :) = [];

%                         % Visualize matrix to facilitate tuning threshold
%                         clear('Features_visual', 'f_visual')
%                         Features_visual = ~isnan(Features_LGG);
%                         f_visual(:,:,1) = zeros(size(Features_visual));
%                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                         f_visual(:,:,3) = zeros(size(Features_visual));
%                         figure(4)
%                         image(f_visual); 
%                         hold on;
%                         title('LGG: original');

% remove features missing more than a threshold of patients
Thesh_f = 0.3; %Best: 0.3
f = sum((isnan(Features_LGG)==1),2);

Features_LGG (f > (Thesh_f * length(Features_LGG(1,:))), :) = [];

Symbols_LGG(f > (Thesh_f * length(Features_LGG(1,:))), :) = [];
SymbolTypes_LGG (f > (Thesh_f * length(Features_LGG(1,:))), :) = [];

%                         % Visualize matrix to facilitate tuning threshold
%                         clear('Features_visual', 'f_visual')
%                         Features_visual = ~isnan(Features_LGG);
%                         f_visual(:,:,1) = zeros(size(Features_visual));
%                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                         f_visual(:,:,3) = zeros(size(Features_visual));
%                         figure(5)
%                         image(f_visual)
%                         hold on;
%                         title('LGG: feature removal');                        
                        
% remove patients missing more than a threshold of features
Thesh_p = 0.0001; %Best: 0.0001
p = sum((isnan(Features_LGG)==1),1);

Features_LGG (:, p > (Thesh_p * length(Features_LGG(:,1)))) = [];
Survival_LGG (:, p > (Thesh_p * length(Features_LGG(:,1)))) = [];
Censored_LGG (:, p > (Thesh_p * length(Features_LGG(:,1)))) = [];
Samples_LGG (:, p > (Thesh_p * length(Features_LGG(:,1)))) = [];


%                         % Visualize matrix to facilitate tuning threshold
%                         clear('Features_visual', 'f_visual')
%                         Features_visual = ~isnan(Features_LGG);
%                         f_visual(:,:,1) = zeros(size(Features_visual));
%                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                         f_visual(:,:,3) = zeros(size(Features_visual));
%                         figure(6)
%                         image(f_visual)
%                         hold on;
%                         title('LGG: patient removal');
                        
% Pack resultant matrix
LGG_Preprocessed.Features = Features_LGG;
LGG_Preprocessed.Symbols = Symbols_LGG;
LGG_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
LGG_Preprocessed.Samples = Samples_LGG; 
LGG_Preprocessed.Survival = Survival_LGG;
LGG_Preprocessed.Censored = Censored_LGG;

%% Intermediate steps ...

% Remove patients missing information on IDH mutation status
unknown1 = isnan(Features_LGG(strcmp(Symbols_LGG, 'IDH1_Mut'),:));
unknown2 = isnan(Features_LGG(strcmp(Symbols_LGG, 'IDH2_Mut'),:));

isUnknown = (unknown1 + unknown2) > 0;

Features_LGG = Features_LGG(:, ~isUnknown);
Survival_LGG = Survival_LGG(:, ~isUnknown);
Censored_LGG = Censored_LGG(:, ~isUnknown);
Samples_LGG = Samples_LGG(:, ~isUnknown);

%% POPULATION 3: Isolate LGG IDHwt patients

%
% NOTE: Use the following thesholds in LGG patient generation (previous
% section) to maximize sample size here.
% Thesh_f = 0.3; Thesh_p = 0.0001;
%

% Get IDHmut status
IDH1mut = Features_LGG(strcmp(Symbols_LGG, 'IDH1_Mut'),:) > min(Features_LGG(strcmp(Symbols_LGG, 'IDH1_Mut'),:));
IDH2mut = Features_LGG(strcmp(Symbols_LGG, 'IDH2_Mut'),:) > min(Features_LGG(strcmp(Symbols_LGG, 'IDH2_Mut'),:));
IDHmut = (IDH1mut + IDH2mut) > 0;

% only keep IDHwt patients
Features_IDHwt = Features_LGG(:, ~IDHmut);
Survival_IDHwt = Survival_LGG(:, ~IDHmut);
Censored_IDHwt = Censored_LGG(:, ~IDHmut);
Samples_IDHwt = Samples_LGG(:, ~IDHmut);

% Pack resultant matrix
IDHwt_Preprocessed.Features = Features_IDHwt;
IDHwt_Preprocessed.Symbols = Symbols_LGG;
IDHwt_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
IDHwt_Preprocessed.Samples = Samples_IDHwt; 
IDHwt_Preprocessed.Survival = Survival_IDHwt;
IDHwt_Preprocessed.Censored = Censored_IDHwt;


%% Intermediate steps ...

% Register patients missing information on 1p, 19q deletion status
unknown1p = isnan(Features_LGG(strcmp(Symbols_LGG, '1p_CNVArm'),:));
unknown19q = isnan(Features_LGG(strcmp(Symbols_LGG, '19q_CNVArm'),:));

% Since this is a copy-number variation, one cannot guarantee that the
% threshold for calling something a chromosomal deletion is if it's < 0,
% but for these two features, it so happens that this is the case (manully
% checked).

del1p = Features_LGG(strcmp(Symbols_LGG, '1p_CNVArm'), :) < 0;
del19q = Features_LGG(strcmp(Symbols_LGG, '19q_CNVArm'), :) < 0; 

del1p(unknown1p==1) = 0; % since NAN values are considered >min
del19q(unknown19q==1) = 0; %since NAN values are considered >min


%% POPULATION 4: Isolate LGG IDHmut-Codel patients

%
% NOTE: Use the following thesholds in LGG patient generation (previous
% section) to maximize sample size here.
% Thesh_f = 0.3; Thesh_p = 0.005;
%

% define samples and features
Features_Codel = Features_LGG;
Survival_Codel = Survival_LGG; 
Censored_Codel = Censored_LGG;
Samples_Codel = Samples_LGG;

% Remove patients with unknown 1p or 19q status or who are not IDHmutCodel
Codel = (del1p + del19q) == 2;

ToDelete = (~IDHmut + unknown1p + unknown19q + ~Codel) > 0;

Features_Codel = Features_Codel(:, ~ToDelete);
Survival_Codel = Survival_Codel(:, ~ToDelete);
Censored_Codel = Censored_Codel(:, ~ToDelete);
Samples_Codel = Samples_Codel(:, ~ToDelete);

% Pack resultant matrix
IDHmutCodel_Preprocessed.Features = Features_Codel;
IDHmutCodel_Preprocessed.Symbols = Symbols_LGG;
IDHmutCodel_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
IDHmutCodel_Preprocessed.Samples = Samples_Codel; 
IDHmutCodel_Preprocessed.Survival = Survival_Codel;
IDHmutCodel_Preprocessed.Censored = Censored_Codel;


%% POPULATION 5: Isolate LGG IDHmut-Non-Codel patients

% define samples and features
Features_NonCodel = Features_LGG;
Survival_NonCodel = Survival_LGG; 
Censored_NonCodel = Censored_LGG;
Samples_NonCodel = Samples_LGG;

%
% Delete patients who you know for sure are not non-Codel or cannot
% ascertain.
%
% NOTE:
% If both 1p and 19q deletion is unknown or if one is deleted and the other
% unknown then either non-Codel or cannot tell. But if one is unknown and
% the other is not deleted, then we known it's non-Codel (because at least
% one is not deleted).
%
Unknown1p_del19q = (unknown1p == 1) & (del19q == 1); 
Unknown19q_del1p = (unknown19q == 1) & (del1p == 1);
BothUnknown = (unknown1p + unknown19q) == 2; 

ToDelete = (~IDHmut + Unknown1p_del19q + Unknown19q_del1p + BothUnknown + Codel) > 0;

Features_NonCodel = Features_NonCodel(:, ~ToDelete);
Survival_NonCodel = Survival_NonCodel(:, ~ToDelete);
Censored_NonCodel = Censored_NonCodel(:, ~ToDelete);
Samples_NonCodel = Samples_NonCodel(:, ~ToDelete);


% Pack resultant matrix
IDHmutNonCodel_Preprocessed.Features = Features_NonCodel;
IDHmutNonCodel_Preprocessed.Symbols = Symbols_LGG;
IDHmutNonCodel_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
IDHmutNonCodel_Preprocessed.Samples = Samples_NonCodel; 
IDHmutNonCodel_Preprocessed.Survival = Survival_NonCodel;
IDHmutNonCodel_Preprocessed.Censored = Censored_NonCodel;
