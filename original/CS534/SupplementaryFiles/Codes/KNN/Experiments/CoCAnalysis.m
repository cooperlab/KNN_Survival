%
% SOURCE NOTE:
% Most of this code is based on the script provided by Dr Lee Cooper in
% Intro to BMI (BMI 500) class, lecture 7
%
%


% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')


%% Choose which model to use


%WhichModel = 'Reduced';
%WhichModel = 'GBM';
%WhichModel = 'LGG';
%WhichModel = 'IDHwt';
%WhichModel = 'IDHmutCodel';
WhichModel = 'IDHmutNonCodel';


load ReducedModel_Unnormalized.mat
Features = ReducedModel.Features;
Survival = ReducedModel.Survival +3;
Censored = ReducedModel.Censored;
Symbols = ReducedModel.Symbols;
SymbolTypes = ReducedModel.SymbolTypes;
Samples = ReducedModel.Samples;

% Identify GBM and LGG patients
isGBM1 = Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-glioblastoma multiforme (gbm)_Clinical'),:));
isGBM2 = Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-treated primary gbm_Clinical'),:));
isGBM3 = Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:) > min(Features(strcmp(Symbols, 'histological_type-Is-untreated primary (de novo) gbm_Clinical'),:));
isGBM = (isGBM1 + isGBM2 + isGBM3) > 0;
isLGG = (isGBM1 + isGBM2 + isGBM3) == 0;
% Get IDHmut status
IDH1mut = Features(strcmp(Symbols, 'IDH1_Mut'),:) == 1;
IDH2mut = Features(strcmp(Symbols, 'IDH2_Mut'),:) == 1;
IDHmut = (IDH1mut + IDH2mut) > 0;
% Identify 11/19q codeletions
Del1p = Features(strcmp(Symbols, '1p_CNVArm'),:) < 0;
Del19q = Features(strcmp(Symbols, '19q_CNVArm'),:) < 0;
Codel = (Del1p + Del19q) == 2;


if strcmp(WhichModel, 'GBM') == 1
        
    Features = Features(:, isGBM);
    Survival = Survival(:, isGBM);
    Censored = Censored(:, isGBM);
    Samples = Samples(:, isGBM);

elseif strcmp(WhichModel, 'LGG') == 1
    
    Features = Features(:, isLGG);
    Survival = Survival(:, isLGG);
    Censored = Censored(:, isLGG);
    Samples = Samples(:, isLGG);

elseif strcmp(WhichModel, 'IDHwt') == 1

    isIDHwt = (isLGG + ~IDHmut) == 2;
    
    Features = Features(:, isIDHwt);
    Survival = Survival(:, isIDHwt);
    Censored = Censored(:, isIDHwt);
    Samples = Samples(:, isIDHwt);

elseif strcmp(WhichModel, 'IDHmutCodel') == 1

    isIDHmutCodel = (isLGG + IDHmut + Codel) == 3;
    
    Features = Features(:, isIDHmutCodel);
    Survival = Survival(:, isIDHmutCodel);
    Censored = Censored(:, isIDHmutCodel);
    Samples = Samples(:, isIDHmutCodel);

elseif strcmp(WhichModel, 'IDHmutNonCodel') == 1

    isIDHmutNonCodel = (isLGG + IDHmut + ~Codel) == 3;
    
    Features = Features(:, isIDHmutNonCodel);
    Survival = Survival(:, isIDHmutNonCodel);
    Censored = Censored(:, isIDHmutNonCodel);    
    Samples = Samples(:, isIDHmutNonCodel);

end


%% Cluster-of-Clusters (CoC) Analysis

% remove features with significant zero values
Discard = sum(Features == 0,2) > 0.2 * size(Features, 2);
Features(Discard, :) = [];
Symbols(Discard) = [];

% calculate standard deviation
StDev = std(Features, [], 2);

% sort by COV - decreasing order
[StDev, Order] = sort(StDev, 'descend');

% No of features to use to differentiate patients to use
FeatNo = 500;

% select top 500 genes, log transform, normalize
Subset = Features(Order(1:FeatNo), :);
Subset = log(Subset + 10); %log transformation suppresses outliers
% Next line is the Z-score (to normalize features so they are comparable to
% each other)
Subset = (Subset - mean(Subset, 2) * ones(1, length(Survival))) ./ ...
            (std(Subset, [], 2) * ones(1, length(Survival)));

% generate clustered heatmap %
cgo = clustergram(Subset, 'Colormap', redbluecmap, 'RowLabels',Symbols(Order(1:FeatNo)),'ColumnLabels',Samples);

% cluster and extract patient cluster labels
rng(1);
K = 2; % no of clusters
C = kmeans(Subset.', K);

% survival analysis
[pLRMedian, ChiSq] = LogRank(Survival, Censored, C);
KMPlot(Survival, Censored, C, {'1','2','3'});
hold on
title(['K-M Plot for ',WhichModel,' for K = ', num2str(K)]);
str = {['p-value = ',num2str(pLRMedian)],['ChiSq = ',num2str(ChiSq)]};
dim = [0.15 0.01 0.2 0.2];
annotation('textbox',dim,'String',str,'FitBoxToText','on','LineWidth',0.01);


% compare expression in custers 1/2 and 3/4
Good = (C == 1 | C == 2);
Bad = (C == 3 | C == 4);
[~, p] = ttest2(Features(:,Good)', Features(:, Bad)');
Difference = median(Features(:, Bad), 2) - ...
    median(Features(:, Good), 2);
Score = -log(p) .* sign(Difference');
[Score, Order] = sort(Score, 'descend');
cell2text(Symbols(Order(1:FeatNo)), 'Genes.txt');