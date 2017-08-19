function X_imputed = KNN_Impute(X,K,OutcomeType)

%
% This predicts NAN point labels based on the labels of their
% K-nearest neighbours using euclidian distance ...
% Customized for imputation of missing values with features as rows and
% samples as columns. Allows for NAN-containing inputs.
%
% INPUTS:
% 
% X - matrix containing sample features (features in rows, samples in cols)
% K - number of nearest-neighbours to use
% OutcomeType - string, either 'Classification' or 'Regression'
% 
% WARNING: Input data has to be normalized to have similar scales
%
% OUTPUTS:
%
% X_imputed - X with all NAN values imputed using KNN based on other
% features in X
%

%% Sample Inputs
% 
% clear ; close all ; clc ; 
% 
% N = 100; %sample size
% p = 12; %no of features
% 
% %X = randn(p,N); % continuous input features (use NAN for missing values to be imputed)
% X = randi([0,1],p,N); % binary input features (use NAN for missing values to be imputed)
% 
% % Add NAN values at random to simulate missing data
% pNaN = 0.1; %proportion of NAN values
% for i = 1:(pNaN * N*p)
% X(randi(p),randi(N)) = nan;
% end
% 
% %OutcomeType = 'Classification';
% OutcomeType = 'Regression';
% K = 5; % number of nearest-neighbours to use

%% Impute feature-by-feature

% initialize
X_imputed = X;

Y_IdxMax = length(X(:,1));

for Y_Idx = 1:Y_IdxMax

% Monitor progress
clc
Current_Feature = Y_Idx    
    
% Isolate feature to impute
Xnew = X;
Y = Xnew(Y_Idx,:);
% Impute based on distance using all other features
Xnew(Y_Idx,:) = [];

% indices of points to classify using KNN
UnknownPoints = 1:1:length(Y(1,:));
UnknownPoints = UnknownPoints(isnan(Y)==1); 

% Initializing output
Center_labels = nan(size(UnknownPoints));

Counter = 1;
for P_Center = UnknownPoints

% current point to label
Center = Xnew(:,P_Center);

% surrounding (all other) points
Surround_labels = Y;
Surround_labels(:,P_Center) = [];
Surround_feat = Xnew;
Surround_feat(:,P_Center) = [];
% remove points with missing labels
Keep_point = ~isnan(Surround_labels);
Surround_labels(Keep_point==0) = [];
Surround_feat(:,Keep_point==0) = [];


%% Compare to every other point using non-missing common features 

% initializa distance to each surrounding point
Dist = zeros(2,length(Surround_feat(1,:)));
Dist(2,:) = Surround_labels;

P_SurroundMax = length(Dist);
for P_Surround = 1:P_SurroundMax

    Surround = Surround_feat(:,P_Surround);
    Keep_feat = ~isnan(Center) & ~isnan(Surround);
    
    Center_temp = Center(Keep_feat==1);
    Surround_temp = Surround(Keep_feat==1);
    
    % No of dimensions being compared
    Ndim = length(Center_temp) ./ length(Center); % proportion of total dimensions kept
    
    % Weighted euclidian distance
    Dist(1,P_Surround) = sum(abs(Center_temp - Surround_temp)) ./ Ndim;
end

% Get nearest neighbours
Dist = sortrows(Dist',1);
Dist = Dist(1:K,:);

if strcmp(OutcomeType,'Classification')
    Center_labels(1,Counter) = mode(Dist(:,2));
elseif strcmp(OutcomeType,'Regression')
    Center_labels(1,Counter) = mean(Dist(:,2));
end

Counter = Counter+1;
end

Y_imputed = Y;
Y_imputed(UnknownPoints) = Center_labels;
X_imputed(Y_Idx,:)= Y_imputed;

end

end