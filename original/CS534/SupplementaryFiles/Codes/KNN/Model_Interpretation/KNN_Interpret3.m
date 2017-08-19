%
% Get K-M curves for datasets based on top genomic features' median split
% (KNN Model interpretation)
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

%WhichModel = 'GBMLGG';
%WhichModel = 'GBM';
%WhichModel = 'LGG';
%WhichModel = 'IDHwt';
%WhichModel = 'IDHmutCodel';
WhichModel = 'IDHmutNonCodel';

load 'GBMLGG.Data.mat';
Survival = Survival +3; %add 3 to ignore negative survival
Symbols = cellstr(Symbols);

        % load 'ReducedModel_Unnormalized.mat';
        % Features = ReducedModel.Features;
        % Survival = ReducedModel.Survival +3;
        % Censored = ReducedModel.Censored;
        % Symbols = ReducedModel.Symbols;
        % SymbolTypes = ReducedModel.SymbolTypes;

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

elseif strcmp(WhichModel, 'LGG') == 1
    
    Features = Features(:, isLGG);
    Survival = Survival(:, isLGG);
    Censored = Censored(:, isLGG);

elseif strcmp(WhichModel, 'IDHwt') == 1

    isIDHwt = (isLGG + ~IDHmut) == 2;
    
    Features = Features(:, isIDHwt);
    Survival = Survival(:, isIDHwt);
    Censored = Censored(:, isIDHwt);

elseif strcmp(WhichModel, 'IDHmutCodel') == 1

    isIDHmutCodel = (isLGG + IDHmut + Codel) == 3;
    
    Features = Features(:, isIDHmutCodel);
    Survival = Survival(:, isIDHmutCodel);
    Censored = Censored(:, isIDHmutCodel);

elseif strcmp(WhichModel, 'IDHmutNonCodel') == 1

    isIDHmutNonCodel = (isLGG + IDHmut + ~Codel) == 3;
    
    Features = Features(:, isIDHmutNonCodel);
    Survival = Survival(:, isIDHmutNonCodel);
    Censored = Censored(:, isIDHmutNonCodel);    

end

% remove NAN survival or censorship values
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

[p,N] = size(Features);

%%

% UNPROCESSED (GBMLGG) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % IDH1 mutation
%     SplitFeat = 'IDH1_Mut';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat == 1);
%     Survival_subpop2 = Survival(Feat == 0);
%     Censored_subpop1 = Censored(Feat == 1);
%     Censored_subpop2 = Censored(Feat == 0);

%     % PTEN deletion
%     SplitFeat = 'PTEN_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0);

%     % 10q arm deletion
%     SplitFeat = '10q_CNVArm';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0);

% GBM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % PPP2R1A amplification
%     SplitFeat = 'PPP2R1A_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat > 0);
%     Survival_subpop2 = Survival(Feat <= 0);
%     Censored_subpop1 = Censored(Feat > 0);
%     Censored_subpop2 = Censored(Feat <= 0);    
 
%     % KLK2 amplification
%     SplitFeat = 'KLK2_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat > 0);
%     Survival_subpop2 = Survival(Feat <= 0);
%     Censored_subpop1 = Censored(Feat > 0);
%     Censored_subpop2 = Censored(Feat <= 0);
    
%     % CCNE1 amplification
%     SplitFeat = 'CCNE1_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat > 0);
%     Survival_subpop2 = Survival(Feat <= 0);
%     Censored_subpop1 = Censored(Feat > 0);
%     Censored_subpop2 = Censored(Feat <= 0);   

% LGG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % CDKN2A deletion
%     SplitFeat = 'CDKN2A_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0); 

%     % IDH1 mutation
%     SplitFeat = 'IDH1_Mut';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat == 1);
%     Survival_subpop2 = Survival(Feat == 0);
%     Censored_subpop1 = Censored(Feat == 1);
%     Censored_subpop2 = Censored(Feat == 0);

%     % PTEN deletion
%     SplitFeat = 'PTEN_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0);    

% IDHwt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % CDKN2A deletion
%     SplitFeat = 'CDKN2A_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0); 

%     % 10p arm deletion
%     SplitFeat = '10p_CNVArm';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0);

%     % PTEN deletion
%     SplitFeat = 'PTEN_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0); 

% Codel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % ZNF292 mutation
%     SplitFeat = 'ZNF292_Mut';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat == 1);
%     Survival_subpop2 = Survival(Feat == 0);
%     Censored_subpop1 = Censored(Feat == 1);
%     Censored_subpop2 = Censored(Feat == 0);

%     % PTPRK deletion
%     SplitFeat = 'PTPRK_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0);     

%     % PREX1 protein amplification
%     SplitFeat = 'PREX1_Protein';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat > 0);
%     Survival_subpop2 = Survival(Feat <= 0);
%     Censored_subpop1 = Censored(Feat > 0);
%     Censored_subpop2 = Censored(Feat <= 0);   

% NonCodel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 9p arm deletion
    SplitFeat = '9p_CNVArm';
    Feat = Features(strcmp(Symbols,SplitFeat), :);
    Survival_subpop1 = Survival(Feat < 0);
    Survival_subpop2 = Survival(Feat >= 0);
    Censored_subpop1 = Censored(Feat < 0);
    Censored_subpop2 = Censored(Feat >= 0);

%     % CDKN2A deletion
%     SplitFeat = 'CDKN2A_CNV';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat < 0);
%     Survival_subpop2 = Survival(Feat >= 0);
%     Censored_subpop1 = Censored(Feat < 0);
%     Censored_subpop2 = Censored(Feat >= 0); 

%     % GATA3 protein amplification
%     SplitFeat = 'GATA3_Protein';
%     Feat = Features(strcmp(Symbols,SplitFeat), :);
%     Survival_subpop1 = Survival(Feat > 0);
%     Survival_subpop2 = Survival(Feat <= 0);
%     Censored_subpop1 = Censored(Feat > 0);
%     Censored_subpop2 = Censored(Feat <= 0); 

%%    
Survival_bothpops = [Survival_subpop1'; Survival_subpop2'];
Censored_bothpops = [Censored_subpop1'; Censored_subpop2'];
Labels = ones(length(Survival_bothpops),1);
Labels(length(Survival_subpop1+1):end,1) = 2;

FeatLabel = strsplit(SplitFeat,'_');
FeatLabel1 = char(FeatLabel(1));
FeatLabel2 = char(FeatLabel(2));

[pLRMedian, ChiSq] = LogRank(Survival_bothpops, Censored_bothpops, Labels);
KMPlot(Survival_bothpops,Censored_bothpops,Labels,{'Subpopulation 1','Subpopulation 2'});
hold on
title(['K-M Plot for ',WhichModel,' model, based on ', FeatLabel1, ' ', FeatLabel2]);
str = {['p-value = ',num2str(pLRMedian)],['ChiSq = ',num2str(ChiSq)]};
dim = [0.15 0.01 0.2 0.2];
annotation('textbox',dim,'String',str,'FitBoxToText','on','LineWidth',0.01);
