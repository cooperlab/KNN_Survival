function Y_test = KNN_Survival2(X_test,X_train,Survival_train,Censored_train,K,Beta1)
%
% This predicts survival based on the labels of K-nearest neighbours 
% using weighted euclidian distance and the K-M estimator.
%
% INPUTS: 
% --------
% IMPORTANT: features as rows, samples as columns
% WARNING: Input data has to be normalized to have similar scales
% 
% X_test - testing sample features
% X_train - training sample features
% Survival_train - survival of training sample
% Censored_train - censorship of training sample: 1 = ALIVE, 0 = DEAD
% K - number of nearest-neighbours to use
% Beta1 - shrinkage factor --> higher values indicate less important
%         features
%
% OUTPUTS:
% ---------
% Y_test - predicted survival times of each patient
%
%% Sample Inputs
% clear ; close all ; clc ; 
% 
% N_train = 100; %training sample size
% N_test = 10; %testing sample size
% p = 12; %no of features
% 
% %
% % FOR THE FOLLOWING: features in rows, samples on columns
% %
% X_train = randn(p,N_train); % training features (continuous)
% %X_train = randi([0,1],p,N_train); % training features (binary)
% 
% X_test = randn(p,N_test); % testing features (continuous)
% %X_test = randi([0,1],p,N_test); % testing features (binary)
% 
% Survival_train = randi([1,300],1,N_train); % survival of training sample
% Censored_train = randi([0,1],1,N_train); % censorship of training sample: 1=alive
% 
% K = 15; % number of nearest-neighbours to use
% Beta1 = randi(1,[p,1]);

%% Begin algorithm
% initialize
Y_test = nan(1,length(X_test(1,:)));
P_Center_Max = length(Y_test(1,:));

for P_Center = 1:P_Center_Max

% current point to label
Center = X_test(:,P_Center);

%% Compare to every other point using non-missing common features 
% initializa distance to each surrounding point
Dist = zeros(3,length(X_train(1,:)));
Dist(2,:) = Survival_train;
Dist(3,:) = Censored_train;

P_SurroundMax = length(Dist);
for P_Surround = 1:P_SurroundMax
    Surround = X_train(:,P_Surround);
    
    % Weighted euclidian distance
    %Dist(1,P_Surround) = sqrt(sum((Beta1 .* (Center - Surround)).^2));
    Dist(1,P_Surround) = sum((Beta1.^2) .* (abs(Center - Surround)));
    
end

% Get nearest neighbours
Dist = sortrows(Dist',1);
Dist = Dist(1:K,:);

% get KM estimator for nearest neighbours
Dist = sortrows(Dist,2);
[t,f,~,~] = KM(Dist(:,2), Dist(:,3));

if sum(Dist(:,3)) < length(Dist(:,3))-1
pAUC = sum(diff(t) .* f(1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
elseif sum(Dist(:,3)) >= length(Dist(:,3))-1 %almost all surrounding points are censored
pAUC = 1;
end

Y_test(1,P_Center) = pAUC * max(t); %since f is maximally 1, max possible survival is tmax*1

end

% Sort by predicted survival time
%Y_test(2,:) = 1:P_Center_Max;
%Y_test = sortrows(Y_test',1);
%Y_test = Y_test(:,2)';

end