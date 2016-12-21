clear; close all; clc;

% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')


%% Read in data and initial preprocessing and feature extraction

load 'GBMLGG.Data.mat';

Zscore_First = 0; %z-score at first or for each subpopulation separately?

% convert to better format
SymbolTypes = cellstr(SymbolTypes);
Symbols = cellstr(Symbols);
Samples = (cellstr(Samples))';

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

if Zscore_First == 1
% Z- score standardization of features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,Features_var] = meshgrid(1:length(Features(1,:)), Features_var);
Features = (Features - Features_mean) ./ (Features_var .^ 0.5);
end

% define different feature matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Clinical = Features(strcmp(SymbolTypes,'Clinical'), :);
Mutation = Features(strcmp(SymbolTypes, 'Mutation'), :);
CNVGene = Features(strcmp(SymbolTypes, 'CNVGene'), :);
CNVArm = Features(strcmp(SymbolTypes, 'CNVArm'), :);
Protein = Features(strcmp(SymbolTypes, 'Protein'), :);
mRNA = Features(strcmp(SymbolTypes, 'mRNA'), :);

%                                         % Visualize matrix at current  moment
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(1)
%                                         image(f_visual)

% save original for monitoring what happens as you delete stuff %%%%%%%%%%%
SymbolTypes_original = SymbolTypes;
Symbols_original = Symbols;
Features_original = Features;
Clinical_original = Clinical;
Mutation_original = Mutation;
CNVGene_original = CNVGene;
CNVArm_original = CNVArm;
Protein_original = Protein;
mRNA_original = mRNA;

%% Make decisions

% Delete any patient or feature missing a single value?
Delete_Only = 1;

% Decide which feature sets to process
Process_Clinical = 1;
Process_Mutation = 1;
Process_CNVGene = 1;
Process_CNVArm = 1;
Process_Protein = 1;
Process_mRNA = 1;

% Decide which features to impute
Impute_Clinical = 0;
Impute_Mutation = 0;
Impute_CNVGene = 0;
Impute_CNVArm = 0;
Impute_Protein = 0;
Impute_mRNA = 0;

%% Define thresholds and parameters

% Define thresholds (for patient removal)
Thr_clin_p = 0;
Thr_mut_p = 0.5;
Thr_cnvgene_p = 0.5;
Thr_cnvarm_p = 1;
Thr_protein_p = 0.5;
Thr_mrna_p = 0.5;

% Define thresholds (for features removal)
Thr_clin_f = 0.01;
Thr_mut_f = 1;
Thr_cnvgene_f = 1;
Thr_cnvarm_f = 1;
Thr_protein_f = 0.4;
Thr_mrna_f = 1;

% Define imputation parameters
K_impute = 30;
K_mode = 'Regression';


%% Set default values for certain options 

if Delete_Only == 1
    % Define thresholds (for patient removal)
    Thr_clin_p = 0;
    Thr_mut_p = 0.5;
    Thr_cnvgene_p = 0.5;
    Thr_cnvarm_p = 1;
    Thr_protein_p = 0.5;
    Thr_mrna_p = 0.5;

    % Define thresholds (for features removal)
    Thr_clin_f = 0.01;
    Thr_mut_f = 1;
    Thr_cnvgene_f = 1;
    Thr_cnvarm_f = 1;
    Thr_protein_f = 0.4;
    Thr_mrna_f = 1;    
end

%% CLINICAL - Remove patients and features accordingly (order matters!)

if Process_Clinical == 1

% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clin_f = sum((isnan(Clinical)==1),2);

Clinical (clin_f > (Thr_clin_f * length(Clinical(1,:))), :) = [];

Symbols (clin_f > (Thr_clin_f * length(Clinical(1,:))), :) = [];
SymbolTypes (clin_f > (Thr_clin_f * length(Clinical(1,:))), :) = [];

% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clin_p = sum((isnan(Clinical)==1),1);

Clinical (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Mutation (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
CNVGene (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
CNVArm (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Protein (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
mRNA (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Survival (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Censored (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Samples (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];

% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_Clinical == 1
Clinical = KNN_Impute(Clinical,K_impute,K_mode);
end

%                                         % Visualize matrix at current  moment
%                                         clear('Features_visual', 'f_visual')
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(2)
%                                         image(f_visual)

end


%% MUTATION - Remove patients and features accordingly (order matters!)

if Process_Mutation == 1

% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mut_f = sum((isnan(Mutation)==1),2);

Mutation (mut_f > (Thr_mut_f * length(Mutation(1,:))), :) = [];

feat_Sofar = length(Clinical(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; mut_f > (Thr_mut_f * length(Mutation(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];

% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mut_p = sum((isnan(Mutation)==1),1);

Clinical (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Mutation (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
CNVGene (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
CNVArm (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Protein (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
mRNA (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Survival (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Censored (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Samples (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];

% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_Mutation == 1
Mutation = KNN_Impute(Mutation,K_impute,K_mode);
end

%                                         % Visualize matrix at current  moment
%                                         clear('Features_visual', 'f_visual')
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(3)
%                                         image(f_visual)

end

%% CNVGene - Remove patients and features accordingly (order matters!)

if Process_CNVGene == 1

% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvgene_f = sum((isnan(CNVGene)==1),2);

CNVGene (cnvgene_f > (Thr_cnvgene_f * length(CNVGene(1,:))), :) = [];

feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; cnvgene_f > (Thr_cnvgene_f * length(CNVGene(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];


% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvgene_p = sum((isnan(CNVGene)==1),1);

Clinical (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Mutation (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
CNVGene (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
CNVArm (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Protein (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
mRNA (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Survival (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Censored (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Samples (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];

% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_CNVGene == 1
CNVGene = KNN_Impute(CNVGene,K_impute,K_mode);
end

%                                         % Visualize matrix at current  moment
%                                         clear('Features_visual', 'f_visual')
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(4)
%                                         image(f_visual)

end

%% CNVArm - Remove patients and features accordingly (order matters!)

if Process_CNVArm == 1

% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvarm_f = sum((isnan(CNVArm)==1),2);

CNVArm (cnvarm_f > (Thr_cnvarm_f * length(CNVArm(1,:))), :) = [];

feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1)) + length(CNVGene(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; cnvarm_f > (Thr_cnvarm_f * length(CNVArm(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];


% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvarm_p = sum((isnan(CNVArm)==1),1);

Clinical (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Mutation (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
CNVGene (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
CNVArm (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Protein (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
mRNA (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Survival (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Censored (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Samples (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];

% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_CNVArm == 1
CNVArm = KNN_Impute(CNVArm,K_impute,K_mode);
end

%                                         % Visualize matrix at current  moment
%                                         clear('Features_visual', 'f_visual')
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(5)
%                                         image(f_visual)

end

%% Protein - Remove patients and features accordingly (order matters!)

if Process_Protein == 1 

% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
protein_f = sum((isnan(Protein)==1),2);

Protein (protein_f > (Thr_protein_f * length(Protein(1,:))), :) = [];

feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1)) + length(CNVGene(:,1)) + length(CNVArm(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; protein_f > (Thr_protein_f * length(Protein(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];


% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
protein_p = sum((isnan(Protein)==1),1);

Clinical (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Mutation (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
CNVGene (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
CNVArm (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Protein (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
mRNA (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Survival (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Censored (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Samples (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];

% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_Protein == 1
Protein = KNN_Impute(Protein,K_impute,K_mode);
end

%                                         % Visualize matrix at current  moment
%                                         clear('Features_visual', 'f_visual')
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(6)
%                                         image(f_visual)

end

%% mRNA - Remove patients and features accordingly (order matters!)

if Process_mRNA == 1

% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrna_f = sum((isnan(mRNA)==1),2);

mRNA (mrna_f > (Thr_mrna_f * length(mRNA(1,:))), :) = [];

feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1)) + length(CNVGene(:,1)) + length(CNVArm(:,1)) + length(Protein(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; mrna_f > (Thr_mrna_f * length(mRNA(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];

% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrna_p = sum((isnan(mRNA)==1),1);

Clinical (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Mutation (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
CNVGene (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
CNVArm (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Protein (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
mRNA (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Survival (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Censored (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Samples (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];

% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_mRNA == 1
mRNA = KNN_Impute(mRNA,K_impute,K_mode);
end

%                                         % Visualize matrix at current  moment
%                                         clear('Features_visual', 'f_visual')
%                                         Features_visual = ~isnan([Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA(1:1000,:)]);
%                                         f_visual(:,:,1) = zeros(size(Features_visual));
%                                         f_visual(:,:,2) = 0.75 .* Features_visual;
%                                         f_visual(:,:,3) = zeros(size(Features_visual));
%                                         figure(7)
%                                         image(f_visual)

end

%% Updating final features matrix

Features = [Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA];

%% SURVIVAL AND CENSORED - Remove patients missing survival/censored data

Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Samples(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Samples(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];


%% Pack data

ReducedModel.Features = Features;
ReducedModel.Symbols = Symbols;
ReducedModel.SymbolTypes = SymbolTypes; 
ReducedModel.Survival = Survival;
ReducedModel.Censored = Censored;
ReducedModel.Samples = Samples;

% Z-score standardization
if Zscore_First == 0
Features_mean = mean(Features, 2);
[~,Features_mean] = meshgrid(1:length(Features(1,:)), Features_mean);
Features_std = std(Features');
[~,Features_std] = meshgrid(1:length(Features(1,:)), Features_std');
Features_Zscored = (Features - Features_mean) ./ Features_std;
Features_Zscored(isnan(Features_Zscored)==1) = 0;
ReducedModel.Features = Features_Zscored;
end

%% POPULATION 1: Isolate GBM sub-population

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

% Pack resultant matrix
GBM_Preprocessed.Features = Features_GBM;
GBM_Preprocessed.Symbols = Symbols_GBM;
GBM_Preprocessed.SymbolTypes = SymbolTypes_GBM; 
GBM_Preprocessed.Samples = Samples_GBM; 
GBM_Preprocessed.Survival = Survival_GBM;
GBM_Preprocessed.Censored = Censored_GBM;

if Zscore_First == 0
% Z-score standardization
Features_mean_GBM = mean(Features_GBM, 2);
[~,Features_mean_GBM] = meshgrid(1:length(Features_GBM(1,:)), Features_mean_GBM);
Features_std_GBM = std(Features_GBM');
[~,Features_std_GBM] = meshgrid(1:length(Features_GBM(1,:)), Features_std_GBM');
Features_Zscored_GBM = (Features_GBM - Features_mean_GBM) ./ Features_std_GBM;
Features_Zscored_GBM(isnan(Features_Zscored_GBM)==1) = 0;
GBM_Preprocessed.Features = Features_Zscored_GBM;
end

%% POPULATION 2: Isolate LGG sub-population

% Remove GBM patients

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

% Pack resultant matrix
LGG_Preprocessed.Features = Features_LGG;
LGG_Preprocessed.Symbols = Symbols_LGG;
LGG_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
LGG_Preprocessed.Samples = Samples_LGG; 
LGG_Preprocessed.Survival = Survival_LGG;
LGG_Preprocessed.Censored = Censored_LGG;

if Zscore_First == 0
%Z-score standardization
Features_mean_LGG = mean(Features_LGG, 2);
[~,Features_mean_LGG] = meshgrid(1:length(Features_LGG(1,:)), Features_mean_LGG);
Features_std_LGG = std(Features_LGG');
[~,Features_std_LGG] = meshgrid(1:length(Features_LGG(1,:)), Features_std_LGG');
Features_Zscored_LGG = (Features_LGG - Features_mean_LGG) ./ Features_std_LGG;
Features_Zscored_LGG(isnan(Features_Zscored_LGG)==1) = 0;
LGG_Preprocessed.Features = Features_Zscored_LGG;
end

%% POPULATION 3: Isolate LGG IDHwt patients

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

if Zscore_First == 0
% Z-score standardization
Features_mean_IDHwt = mean(Features_IDHwt, 2);
[~,Features_mean_IDHwt] = meshgrid(1:length(Features_IDHwt(1,:)), Features_mean_IDHwt);
Features_std_IDHwt = std(Features_IDHwt');
[~,Features_std_IDHwt] = meshgrid(1:length(Features_IDHwt(1,:)), Features_std_IDHwt');
Features_Zscored_IDHwt = (Features_IDHwt - Features_mean_IDHwt) ./ Features_std_IDHwt;
Features_Zscored_IDHwt(isnan(Features_Zscored_IDHwt)==1) = 0;
IDHwt_Preprocessed.Features = Features_Zscored_IDHwt;
end

%% POPULATION 4: Isolate LGG IDHmut-Codel patients

% Get 1p/19q co-deletion status
% Since this is a copy-number variation, one cannot guarantee that the
% threshold for calling something a chromosomal deletion is < 0 after Z-scoring,
% but for these two features, it so happens that this is the case (manually
% checked).
Del1p = Features_LGG(strcmp(Symbols_LGG, '1p_CNVArm'),:) < 0;
Del19q = Features_LGG(strcmp(Symbols_LGG, '19q_CNVArm'),:) < 0;
Codel = (Del1p + Del19q) == 2;

IDHmut_Codel = (IDHmut + Codel) == 2;

% only keep codel patients
Features_Codel = Features_LGG(:, IDHmut_Codel);
Survival_Codel = Survival_LGG(:, IDHmut_Codel);
Censored_Codel = Censored_LGG(:, IDHmut_Codel);
Samples_Codel = Samples_LGG(:, IDHmut_Codel);

% Pack resultant matrix
IDHmutCodel_Preprocessed.Features = Features_Codel;
IDHmutCodel_Preprocessed.Symbols = Symbols_LGG;
IDHmutCodel_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
IDHmutCodel_Preprocessed.Samples = Samples_Codel; 
IDHmutCodel_Preprocessed.Survival = Survival_Codel;
IDHmutCodel_Preprocessed.Censored = Censored_Codel;

if Zscore_First == 0
% Z-score standardization
Features_mean_Codel = mean(Features_Codel, 2);
[~,Features_mean_Codel] = meshgrid(1:length(Features_Codel(1,:)), Features_mean_Codel);
Features_std_Codel = std(Features_Codel');
[~,Features_std_Codel] = meshgrid(1:length(Features_Codel(1,:)), Features_std_Codel');
Features_Zscored_Codel = (Features_Codel - Features_mean_Codel) ./ Features_std_Codel;
Features_Zscored_Codel(isnan(Features_Zscored_Codel)==1) = 0;
IDHmutCodel_Preprocessed.Features = Features_Zscored_Codel;
end

%% POPULATION 5: Isolate LGG IDHmut-Non-Codel patients

IDHmut_NonCodel = (IDHmut == 1) & (Codel == 0);

% only keep Non-codel patients
Features_NonCodel = Features_LGG(:, IDHmut_NonCodel);
Survival_NonCodel = Survival_LGG(:, IDHmut_NonCodel);
Censored_NonCodel = Censored_LGG(:, IDHmut_NonCodel);
Samples_NonCodel = Samples_LGG(:, IDHmut_NonCodel);

% Pack resultant matrix
IDHmutNonCodel_Preprocessed.Features = Features_NonCodel;
IDHmutNonCodel_Preprocessed.Symbols = Symbols_LGG;
IDHmutNonCodel_Preprocessed.SymbolTypes = SymbolTypes_LGG; 
IDHmutNonCodel_Preprocessed.Samples = Samples_NonCodel; 
IDHmutNonCodel_Preprocessed.Survival = Survival_NonCodel;
IDHmutNonCodel_Preprocessed.Censored = Censored_NonCodel;

if Zscore_First == 0
% Z-score standardization
Features_mean_NonCodel = mean(Features_NonCodel, 2);
[~,Features_mean_NonCodel] = meshgrid(1:length(Features_NonCodel(1,:)), Features_mean_NonCodel);
Features_std_NonCodel = std(Features_NonCodel');
[~,Features_std_NonCodel] = meshgrid(1:length(Features_NonCodel(1,:)), Features_std_NonCodel');
Features_Zscored_NonCodel = (Features_NonCodel - Features_mean_NonCodel) ./ Features_std_NonCodel;
Features_Zscored_NonCodel(isnan(Features_Zscored_NonCodel)==1) = 0;
IDHmutNonCodel_Preprocessed.Features = Features_Zscored_NonCodel;
end