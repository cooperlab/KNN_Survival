clear; close all; clc; 
addpath('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/glmnet_matlab')

%% Define params

sites = {'GBMLGG', 'KIPAN'};
dtypes = {'Integ', 'Gene'};

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/';
result_path = strcat(base_path, 'Results/13_22Oct2017/');

%% go through sites and dtypes

siteidx = 1; dtypeidx = 1; 
% for siteidx = 1:length(sites)
%     for dtypeidx = 1:length(dtypes)

%% load data and split indices

site = sites{siteidx};
dtype = dtypes{dtypeidx};

dpath = strcat(base_path, 'Data/SingleCancerDatasets/', site, '/', ...
               site, '_', dtype, '_Preprocessed');
          
Data = load(strcat(dpath, '.mat'));
splitIdxs = load(strcat(dpath, '_splitIdxs.mat'));

%% 
% NOTE: Cox elastic net models contain two hyparparameters, 
% 1- ALPHA which controls the overall degree of regularization and 
% 2- the mixture coefficient LAMBDA which controls the balance between L2 
%    and L1 norm penalties.

% turn off warnings
warning('off','all')

% range of ALPHA for elastic net
elasticnet = [1 .7 .5 .3 .1 .0];

% initialize
n_folds = length(splitIdxs.test);
c = zeros(n_folds, 1); % testing c-index

%% go through folds

fold = 1;
% for fold = 1:n_folds

max = 0;
opts.intr = false;
opts.maxit = 100;
chosen_lambda = zeros(length(elasticnet), 1);

%% go through alphas

i = 1;
%for i=1:length(elasticnet)

opts.alpha = elasticnet(i);
ci = 0;
max_cindex = 0;

fprintf('\ni = %d \n', i)
fprintf('fold = %d \n\n', fold)

%% Going through the various LAMBDAS for elastic net

opts.nlambda = 20;

% Separate various sets

idxs_train = splitIdxs.train{1, fold} + 1;
idxs_valid = splitIdxs.valid(fold, :) + 1;
idxs_test = splitIdxs.test{1, fold} + 1;

if dtype == 'Integ'
    X_train = Data.Integ_X(idxs_train, :);
else
    X_train = Data.Gene_X(idxs_train, :);
end

% notice that glmnet takes observed event (1 - censorship status)
Y_train = [Data.Survival(:, idxs_train); 1 - Data.Censored(:, idxs_train)]'; 

%%%%% W A R N I N G !!!! %%%%%%%%%%%%%%%%%
% just for prototyping
X_train = X_train(1:50, 1:10);
Y_train = Y_train(1:50, :);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% fit model
fit = glmnet(X_train, Y_train, 'cox', opts);  

% for b = 1:size(fit.lambda)
%     % Getting the c-index for current lambda
%     cindex = cIndex(fit.beta(:, b), X_CV, T_CV, C_CV);
%     if (max_cindex < cindex)           
%         lambda = fit.lambda(b);
%         max_cindex = cindex;
%     end
% end
% % ADDING optimal lambda and corresponding ci for current shuffle
% chosen_lambda(i) = chosen_lambda(i) + lambda;
% ci = ci + max_cindex;