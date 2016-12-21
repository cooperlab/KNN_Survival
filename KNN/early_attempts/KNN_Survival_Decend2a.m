function [sigma_star,Progress] = KNN_Survival_Decend2a(X_valid,X_train,Alive_train,Alive_valid,K,Beta,Filters,Gamma,Pert,Conv_Thresh,sigma_init,SaveProgress)

%
% This determines the optimum value for SIGMA using gradient descent
% WARNING: make sure that ..
%           1- Features are normalized to have similar scales
%           2- Features in rows, samples in columns
%           3- Same time range for Alive_train and Alive_valid (starting
%              from (minimum_survival_time - 1), and that minimum survival
%              time is 2 days.
%

%% Sample Inputs
% clear ; close all ; clc ; 
% 
% N_train = 100; %training sample size
% N_valid = 100; %testing sample size
% p = 12; %no of features
% 
% %
% % FOR THE FOLLOWING: features in rows, samples on columns
% %
% X_train = randn(p,N_train); % training features (continuous)
% X_valid = randn(p,N_valid); % testing features (continuous)
% 
% t_min = 1;
% t_max = 300;
% Survival_train = randi([t_min,t_max],1,N_train); % survival of training sample
% Survival_valid = randi([t_min,t_max],1,N_valid); % survival of training sample
% Censored_train = randi([0,1],1,N_train); % censorship of training sample: 1=alive
% Censored_valid = randi([0,1],1,N_valid); % censorship of training sample: 1=alive
% 
% % Convert outcome from survival to alive/dead status using time indicator
% Alive_train = TimeIndicator(Survival_train,Censored_train,t_min,t_max);
% Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);
% 
% K = 30; % number of nearest-neighbours to use
% Beta = randi(1,[p,1]); %shrinkage factor for each feature
% 
% %Filters = 'Euclidian';
% %Filters = 'Gaussian';
% Filters = 'Both';
% %Filters = 'None';
% 
% Gamma = 10; %learning rate
% Pert = 0.1; %this controls how much to perturb beta to get a feeling for gradient
% Conv_Thresh = 0.01; %convergence threshold for sigma
% sigma_init = 3; %initial sigma value
%
%% Begin algorithm

% initialize sigma
sigma0 = sigma_init; %sigma of gaussian filter (lower values result in more emphasis on closes neighbours)

%% Get cost for current sigma
% predicted survival
Alive_valid_hat0 = KNN_Survival3(X_valid,X_train,Alive_train,K,Beta,Filters,sigma0);
% cost over each time
Cost0 = (Alive_valid - Alive_valid_hat0).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost0);
Cost0(Cost_nan == 1) = 0;
Cost0 = sum(Cost0);
Cost0 = Cost0 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

sigma_star = sigma0;
Cost_star = Cost0;

%%
% initialize loop
step = 0;
Convergence = 0;

while Convergence == 0

step = step + 1;    
    
% Uncomment the following to monitor progress
% clc
% step    
% sigma0
% sigma_star
% COST0 = mean(Cost0)
% COST_STAR = mean(Cost_star)   

if strcmp(SaveProgress,'SaveProgress') ==1
Progress.step(step,1) = step;
Progress.sigma(step,1) = sigma0;
Progress.sigma_star(step,1) = sigma_star;
Progress.cost(step,1) = mean(Cost0);
Progress.cost_star(step,1) = mean(Cost_star);
end

%% Perturb sigma and calculate new cost
sigma1 = sigma0 + Pert;

% predicted survival
Alive_valid_hat1 = KNN_Survival3(X_valid,X_train,Alive_train,K,Beta,Filters,sigma1);
% cost over each time
Cost1 = (Alive_valid - Alive_valid_hat1).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost1);
Cost1(Cost_nan == 1) = 0;
Cost1 = sum(Cost1);
Cost1 = Cost1 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Batch Gradient descent

% Calculate gradient for each sample
deltaCost = Cost1 - Cost0;
deltaSigma = sigma1 - sigma0;
Gradient = deltaCost ./ deltaSigma;

% Calculate average gradient
Gradient = mean(Gradient);

% Update sigma
%sigma_New = sigma0 - (Gamma .* Gradient);
sigma0 = sigma0 - (Gamma .* Gradient);

%% Get cost for current sigma

% predicted survival
Alive_valid_hat0 = KNN_Survival3(X_valid,X_train,Alive_train,K,Beta,Filters,sigma0);
% cost over each time
Cost0 = (Alive_valid - Alive_valid_hat0).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost0);
Cost0(Cost_nan == 1) = 0;
Cost0 = sum(Cost0);
Cost0 = Cost0 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% update optimum sigma

if mean(Cost0) < mean(Cost_star)
    sigma_star = sigma0;
    Cost_star = Cost0;
end

%% determine convergence

% determine if convergence reached
if abs(mean(deltaCost)) <= Conv_Thresh
    Convergence = 1; 
end

% break if cannot reach convergence
if step > 100
    Convergence = 1;
end

end


end