clear; close all; clc;
 
% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')

 
% turn off warnings
% warning('off','all')


%% Read in data and initial preprocessing and feature extraction
 
load 'BRCA.Data.mat';
 
% convert to better format
SymbolTypes = cellstr(SymbolTypes);
Symbols = cellstr(Symbols);

%define a basic model containing IDH mutations, age and chromosomes 1p/19q
x1a = Features(strcmp(cellstr(Symbols),...
    'age_at_initial_pathologic_diagnosis_Clinical'), :);
x1b = Features(strcmp(cellstr(Symbols),...
    'histological_type-Is-mucinous carcinoma_Clinical'), :);
x1c = Features(strcmp(cellstr(Symbols),...
    'pathologic_stage-Is-stage iv_Clinical'), :);
x1d = Features(strcmp(cellstr(Symbols),...
    'number_of_lymphnodes_positive_by_he_Clinical'), :);
x1e = Features(strcmp(cellstr(Symbols),...
    'pathologic_m-Is-m1'), :);

x2 = Features(strcmp(cellstr(Symbols), 'ERBB2_Mut'), :);
x3 = Features(strcmp(cellstr(Symbols), 'ALDH4A1_mRNA'), :);
x4 = Features(strcmp(cellstr(Symbols), 'BBC3_mRNA'), :);
x5 = Features(strcmp(cellstr(Symbols), 'WISP1_mRNA'), :);
x6 = Features(strcmp(cellstr(Symbols), 'TGFB3_mRNA'), :);
x7 = Features(strcmp(cellstr(Symbols), 'RAB6B_mRNA'), :);
x8 = Features(strcmp(cellstr(Symbols), 'MMP9_mRNA'), :);
x9 = Features(strcmp(cellstr(Symbols), 'OXCT1_mRNA'), :);

S1 = Features(strcmp(cellstr(Symbols), 'GSTM3_mRNA'), :); %pos
S2 = Features(strcmp(cellstr(Symbols), 'GNAZ_mRNA'), :); % neg
S3 = Features(strcmp(cellstr(Symbols), 'FLT1_mRNA'), :); % neg
S4 = Features(strcmp(cellstr(Symbols), 'EXT1_mRNA'), :); % neg
S5 = Features(strcmp(cellstr(Symbols), 'STK32B_mRNA'), :); % pos
S6 = Features(strcmp(cellstr(Symbols), 'ECT2_mRNA'), :); % neg
S7 = Features(strcmp(cellstr(Symbols), 'GMPS_mRNA'), :); % neg
S8 = Features(strcmp(cellstr(Symbols), 'CDC42BPA_mRNA'), :); % neg


%% Final feature matrix

Features = [x1a;x1b;x1c;x1d;x1e;x2;x3;x4;x5;x6;x7;x8;x9; ...
            S1;S2;S3;S4;S5;S6;S7;S8];

%% define core set of samples that have all basic features, survival, censoring
Keep1 = ~isnan(Survival) & ~isnan(Censored) ...
    & (sum(isnan(Features), 1) == 0);
Features = Features(:, Keep1);
Survival = Survival(Keep1);
Censored = Censored(Keep1);

%% More preprocessing

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
 
Features_mean(Features_var == 0, :) = [];
Features_var(Features_var == 0, :) = [];
 
% Z- score standardization of features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,Features_var] = meshgrid(1:length(Features(1,:)), Features_var);
Features = (Features - Features_mean) ./ (Features_var .^ 0.5);
        
%% Pack data

BasicModel.Features = Features;
BasicModel.Survival = Survival;
BasicModel.Censored = Censored;