function [Beta_star,Progress] = KNN_Survival_Decend2b(X_valid,X_train,Alive_train,Alive_valid,K,Beta_init,Filters,Gamma,Pert,Conv_Thresh,sigma,SaveProgress)

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
% Beta_init = randi(1,[p,1]); %shrinkage factor for each feature
% 
% %Filters = 'Euclidian';
% %Filters = 'Gaussian';
% Filters = 'Both';
% %Filters = 'None';
% 
% sigma = 10; %sigma of gaussian filter (lower values result in more emphasis on closes neighbours)
% Gamma = 15; %learning rate
% Pert = 5; %this controls how much to perturb beta to get a feeling for gradient
% Conv_Thresh = 0.0001; %convergence threshold


%% Begin algorithm

% initialize beta
Beta0 = Beta_init; %shrinkage factor for each feature
    
%% Get cost for initialized beta
% predicted survival
Alive_valid_hat0 = KNN_Survival3(X_valid,X_train,Alive_train,K,Beta0,Filters,sigma);
% cost over each time
Cost0 = (Alive_valid - Alive_valid_hat0).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost0);
Cost0(Cost_nan == 1) = 0;
Cost0 = sum(Cost0);
Cost0 = Cost0 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Start gradient descent till convergence

Beta_star = Beta0;
Cost_star = Cost0;

step = 0;
Convergence = 0;
while Convergence == 0 

step = step + 1;    

% Uncomment the following to monitor progress
clc
step
%BETA0 = Beta0'
%BETA_STAR = Beta_star'
COST0 = mean(Cost0)
COST_STAR = mean(Cost_star)

if strcmp(SaveProgress,'SaveProgress') ==1
Progress.step(step,1) = step;
Progress.Beta(step,:) = Beta0';
Progress.Beta_star(step,:) = Beta_star';
Progress.cost(step,1) = mean(Cost0);
Progress.cost_star(step,1) = mean(Cost_star);
end

%% Find gradient with respect to each component in beta

Gradient = zeros(size(Beta0));

for i = 1:length(Beta0)


%% Perturb beta component and calculate new cost

Beta1 = Beta0;
Beta1(i,1) = Beta0(i,1) + Pert;

% predicted survival
Alive_valid_hat1 = KNN_Survival3(X_valid,X_train,Alive_train,K,Beta1,Filters,sigma);
% cost over each time
Cost1 = (Alive_valid - Alive_valid_hat1).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost1);
Cost1(Cost_nan == 1) = 0;
Cost1 = sum(Cost1);
Cost1 = Cost1 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Calculate gradient

deltaCost = Cost1 - Cost0;
deltaBeta = Beta1(i,1) - Beta0(i,1);
gradient = deltaCost ./ deltaBeta;

% Calculate average gradient
Gradient(i,1) = mean(gradient);

end


%% Update beta
Beta0 = Beta0 - (Gamma .* Gradient);

%% Get cost for new beta

% predicted survival
Alive_valid_hat1 = KNN_Survival3(X_valid,X_train,Alive_train,K,Beta0,Filters,sigma);
% cost over each time
Cost1 = (Alive_valid - Alive_valid_hat1).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost1);
Cost1(Cost_nan == 1) = 0;
Cost1 = sum(Cost1);
Cost1 = Cost1 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Get difference in cost and act accordingle

% update optimum beta
if mean(Cost1) < mean(Cost_star)
    Beta_star = Beta0;
    Cost_star = Cost1;
end

% determine if convergence reached
if abs( mean(Cost1) - mean(Cost0) ) > Conv_Thresh
    Cost0 = Cost1;
elseif abs( mean(Cost1) - mean(Cost0) ) <= Conv_Thresh
    Convergence = 1; 
end

% exit if takes too long to converge
if step > 40
    Convergence = 1;
end

end

end