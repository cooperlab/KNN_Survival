% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')

%% Load data

load 'ReducedModel.mat';
Features = ReducedModel.Features;
Survival = ReducedModel.Survival +3; %add 3 to ignore negative survival
Censored = ReducedModel.Censored;

% remove NAN survival or censorship values
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

[p,N] = size(Features);

%% Find feature correlations (all except mRNA) and plot

% Get subset excluding mRNA
Features_subset = Features(1:398,:);

% Get correlations
CorrMat = corr(Features(1:398,:)',Features(1:398,:)');

% Plot Correlations
% CorrMat_im(:,:,1) = abs(CorrMat);
% CorrMat_im(:,:,2) = zeros(size(CorrMat));
% CorrMat_im(:,:,3) = zeros(size(CorrMat));
% image(CorrMat_im)

%% Choose subsets of decreasing correlation

% get lower triangular part of CorrMat
CorrMat = tril(CorrMat);
% remove self-correlation
CorrMat(CorrMat > 0.99999) = 0;
% get absolut correlations
CorrMat = abs(CorrMat);

% get quantile limits
c = CorrMat(CorrMat ~= 0);
qlim0 = quantile(c,0.004);
qlim5a = quantile(c,0.050);
qlim5b = quantile(c,0.054);
qlim10a = quantile(c,0.100);
qlim10b = quantile(c,0.104);
qlim15a = quantile(c,0.150);
qlim15b = quantile(c,0.154);
qlim20a = quantile(c,0.200);
qlim20b = quantile(c,0.204);
qlim25a = quantile(c,0.250);
qlim25b = quantile(c,0.254);
qlim50a = quantile(c,0.500);
qlim50b = quantile(c,0.504);
qlim75a = quantile(c,0.750);
qlim75b = quantile(c,0.754);
qlim95a = quantile(c,0.950);
qlim95b = quantile(c,0.955);
qlim100 = quantile(c,0.996);

% Divide into subsets
CorrMat_0 = CorrMat; CorrMat_0(CorrMat >= qlim0) = 0;
CorrMat_5 = CorrMat; CorrMat_5(CorrMat < qlim5a | CorrMat >= qlim5b) = 0;
CorrMat_10 = CorrMat; CorrMat_10(CorrMat < qlim10a | CorrMat >= qlim10b) = 0;
CorrMat_15 = CorrMat; CorrMat_15(CorrMat < qlim15a | CorrMat >= qlim15b) = 0;
CorrMat_20 = CorrMat; CorrMat_20(CorrMat < qlim20a | CorrMat >= qlim20b) = 0;
CorrMat_25 = CorrMat; CorrMat_25(CorrMat < qlim25a | CorrMat >= qlim25b) = 0;
CorrMat_50 = CorrMat; CorrMat_50(CorrMat < qlim50a | CorrMat >= qlim50b) = 0;
CorrMat_75 = CorrMat; CorrMat_75(CorrMat < qlim75a | CorrMat >= qlim75b) = 0;
CorrMat_95 = CorrMat; CorrMat_95(CorrMat < qlim95a | CorrMat >= qlim95b) = 0;
CorrMat_100 = CorrMat; CorrMat_100(CorrMat < qlim100) = 0;

% Plot subset
% figure(1)
% CorrMat_subset1_im(:,:,1) = 0.75 .* CorrMat_subset1 > 0;
% CorrMat_subset1_im(:,:,2) = zeros(size(CorrMat));
% CorrMat_subset1_im(:,:,3) = zeros(size(CorrMat));
% image(CorrMat_subset1_im)


%% Get actual feature subsets

% create X and Y grid
[X,Y] = meshgrid(1:length(CorrMat),1:length(CorrMat));
X_q0 = X(CorrMat_0 ~= 0);
Y_q0 = Y(CorrMat_0 ~= 0);
X_q5 = X(CorrMat_5 ~= 0);
Y_q5 = Y(CorrMat_5 ~= 0);
X_q10 = X(CorrMat_10 ~= 0);
Y_q10 = Y(CorrMat_10 ~= 0);
X_q15 = X(CorrMat_15 ~= 0);
Y_q15 = Y(CorrMat_15 ~= 0);
X_q20 = X(CorrMat_20 ~= 0);
Y_q20 = Y(CorrMat_20 ~= 0);
X_q25 = X(CorrMat_25 ~= 0);
Y_q25 = Y(CorrMat_25 ~= 0);
X_q50 = X(CorrMat_50 ~= 0);
Y_q50 = Y(CorrMat_50 ~= 0);
X_q75 = X(CorrMat_75 ~= 0);
Y_q75 = Y(CorrMat_75 ~= 0);
X_q95 = X(CorrMat_95 ~= 0);
Y_q95 = Y(CorrMat_95 ~= 0);
X_q100 = X(CorrMat_100 ~= 0);
Y_q100 = Y(CorrMat_100 ~= 0);

% get unique feature indices
Feat0 = unique([X_q0,Y_q0]);
Feat5 = unique([X_q5,Y_q5]);
Feat10 = unique([X_q10,Y_q10]);
Feat15 = unique([X_q15,Y_q15]);
Feat20 = unique([X_q20,Y_q20]);
Feat25 = unique([X_q25,Y_q25]);
Feat50 = unique([X_q50,Y_q50]);
Feat75 = unique([X_q75,Y_q75]);
Feat95 = unique([X_q95,Y_q95]);
Feat100 = unique([X_q100,Y_q100]);


% make sure to use the same number of features in each subset
mindim = min(length(Feat0(:)),length(Feat5(:)));
mindim = min(mindim,length(Feat10(:)));
mindim = min(mindim,length(Feat15(:)));
mindim = min(mindim,length(Feat20(:)));
mindim = min(mindim,length(Feat25(:)));
mindim = min(mindim,length(Feat50(:)));
mindim = min(mindim,length(Feat75(:)));
mindim = min(mindim,length(Feat95(:)));
mindim = min(mindim,length(Feat100(:)));

Feat0 = Feat0(1:mindim,:);
Feat5 = Feat5(1:mindim,:);
Feat10 = Feat10(1:mindim,:);
Feat15 = Feat15(1:mindim,:);
Feat20 = Feat20(1:mindim,:);
Feat25 = Feat25(1:mindim,:);
Feat50 = Feat50(1:mindim,:);
Feat75 = Feat75(1:mindim,:);
Feat95 = Feat95(1:mindim,:);
Feat100 = Feat100(1:mindim,:);