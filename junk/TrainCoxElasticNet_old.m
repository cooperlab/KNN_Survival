function [c, HR] = TrainCoxElasticNet(X_original,T_original,C_original,shuffles,MONITOR)

%
% *************************************************************************
% *** Cox-Elastic Net regression ******************************************
% *************************************************************************
% Disclaimer:
% This code was originally written by Safoora and has been adapted for 
% my own project.
%
% It gets c-index using train/valid/test 60/20/20 split
%
% Inputs:
% -------
% X_original - data (patients in rows, features in columns)
% T_original - time to event (vector)
% C_original - right-censorship (1= alive, 0=dead)
% shuffles (optional) - no of shuffles, default is 10
% MONITOR (optional) - monitor progress?, default is false
%

% Set default parameters if not provided
switch nargin
    case 3
        shuffles = 10;
        MONITOR = true;
    case 4
        MONITOR = true;
end

% randomly generating seed for rng()
Seed = randi(10);

%% Randomly generate data to test script

% clear; close all; clc;
% shuffles = 10; % no of shuffles
% MONITOR = true;
% 
% % original
% X_original = randn([1000, 30]);
% T_original = randi([1,300], [1000,1]);
% C_original = randi([0, 1], [1000,1]);

%% Initialize

addpath('/home/mohamed/Desktop/CooperLab_Research/TransferL_Project/Codes/supporting/glmnet_matlab')
% turn off warnings
warning('off','all')

% NOTE: Cox elastic net models contain two hyparparameters, 
% 1- ALPHA which controls the overall degree of regularization and 
% 2- the mixture coefficient LAMBDA which controls the balance between L2 
%    and L1 norm penalties.

% ensure there are no negative survival times
T_original = T_original - min(T_original) +1;

% range of ALPHA for elastic net
elasticnet = [1 .7 .5 .3 .1 .0];

% initialize
c = zeros(shuffles, 1); % testing c-index
max = 0;
opts.intr = false;
opts.maxit = 100;
chosen_lambda = zeros(length(elasticnet), 1);


%% Going through various ALPHAS (and LAMBDAS)

if (MONITOR == true)
    fprintf('Getting optimal alpha and lambda for glmnet \n\n')
end

for i=1:length(elasticnet)
    
    % Getting original order
    % NOTE: each time we get initial order so that when we use a set random
    % seed (rng) we get the EXACT same shuffles to prevent mixing of
    % training and testing sets
    T = T_original;
    C = C_original;
    X = X_original;
    
    % IMPORTANT!!! This ensures the SAME shuffles to maintain separation 
    % of training and testing
    rng(Seed); 
    
    opts.alpha = elasticnet(i);
    ci = 0;
    max_cindex = 0;
    
    % shuffling multiple times
    for sh = 1:shuffles

	if (MONITOR == true)
            fprintf('\ni = %d \n', i)
            fprintf('sh = %d \n\n', sh)
	end

        m = size(X, 1);
        order = randperm(m);
        T = T(order);
        C = C(order);
        X = X(order,:);
        
        foldsize = floor(20 * m / 100);

        % training (60%) --------------------------------------------------
        X_train = X(2 * foldsize + 1:end, :);
        T_train = T(2 * foldsize + 1:end);
        C_train = C(2 * foldsize + 1:end);
        %notice the flipping of censorship sign for glmnet
        Y_train = [T_train, -C_train+1]; 
        
        % validation (20%) ------------------------------------------------
        X_CV = X(foldsize + 1:2 * foldsize, :);
        T_CV = T(foldsize + 1:2 * foldsize);
        C_CV = C(foldsize + 1:2 * foldsize);
        
        % Going through the various LAMBDAS for elastic net
        opts.nlambda = 20;
        fit = glmnet(X_train, Y_train, 'cox', opts);  
        
        for b = 1:size(fit.lambda)
            % Getting the c-index for current lambda
            cindex = cIndex(fit.beta(:, b), X_CV, T_CV, C_CV);
            if (max_cindex < cindex)           
                lambda = fit.lambda(b);
                max_cindex = cindex;
            end
        end
        % ADDING optimal lambda and corresponding ci for current shuffle
        chosen_lambda(i) = chosen_lambda(i) + lambda;
        ci = ci + max_cindex;
    end
    
    
    % Getting average ci for current alpha
    ci = ci / shuffles;
    
    % save alpha if it is the optimum
    if (ci > max)
        chosen_enc = i;
        max = ci;
    end  
end

% Getting optimal lambda
chosen_lambda = chosen_lambda / shuffles;

if (MONITOR == true)
    fprintf('\noptimal alpha = %d \n', chosen_enc)
end


%% Training using optimal parameters

if (MONITOR == true)
    fprintf('Now training using optimal parameters \n\n')
end

% Getting original order
% NOTE: each time we get initial order so that when we use a set random
% seed (rng) we get the EXACT same shuffles to prevent mixing of
% training and testing sets
T = T_original;
C = C_original;
X = X_original;

% IMPORTANT!!! This ensures the SAME shuffles to maintain separation 
% of training and testing
rng(Seed);

for sh = 1:shuffles
    
    fprintf('Going through \n')
    sh
    
    m = size(X, 1);
    order = randperm(m);
    T = T(order);
    C = C(order);
    X = X(order,:);
    
    foldsize = floor(20 * m / 100);

    % testing (20%) -------------------------------------------------------
    X_test = X(1:foldsize, :);
    T_test = T(1:foldsize);
    C_test = C(1:foldsize);

    % Now using the two sets originally separated into training and
    % validation for training the final model
    % training (80%) -------------------------------------------------------
    X_train = X(1 * foldsize + 1:end, :);
    T_train = T(1 * foldsize + 1:end);
    C_train = C(1 * foldsize + 1:end);
    %notice the flipping of censorship sign for glmnet
    Y_train = [T_train, -C_train+1];

    % training using the optimal parameters on training/validation set
    opts.alpha = elasticnet(chosen_enc);
    opts.lambda = chosen_lambda(chosen_enc);
    opts.nlambda = 1;
    fit = glmnet(X_train, Y_train, 'cox', opts);

    % now getting the testing error
    c(sh) = cIndex(fit.beta(:, 1), X_test, T_test, C_test);

    %if (MONITOR == true)
    %fprintf('Finished shuffle : %d \n', sh)
    %fprintf('c-index = %d \n\n', c(sh))
    %end
end

if (MONITOR == true)
    fprintf('\nc-indices : \n')
    c
    fprintf('mean c-index = %d \n', mean(c))
    fprintf('std of c-indices = %d \n\n', std(c))
    fprintf('Training on full dataset to get Hazard Ratios... \n')
end


%% Train on full dataset to get hazard ratios for model interpretation

%notice the flipping of censorship sign for glmnet
Y_original = [T_original, -C_original+1];

opts.alpha = elasticnet(chosen_enc);
opts.lambda = chosen_lambda(chosen_enc);
opts.nlambda = 1;
fit = glmnet(X_original, Y_original, 'cox', opts);
HR = exp(fit.beta(:, 1));


%%
% turn warnings back on
warning('on','all')

end