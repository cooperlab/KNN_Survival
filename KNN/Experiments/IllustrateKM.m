%
% Assess performance of basic model using training/validation/testing
% approach with shuffling - KNN using area under curve * tmax (to get
% ACTUAL survival times)
%
% NOTE: Codes used here are a combination of original codes and codes
% provided in the "PerformanceExample.m" script provided by Dr Lee Cooper
% in CS-534 class
%
% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')

% turn off warnings
% warning('off','all')

%% Using basic model as provided by Dr Lee (PerformanceExample.m) for a starter

% read in data and extract minimal features
load 'BasicModel.mat';

Features = BasicModel.Features;
Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
Censored = BasicModel.Censored;

[p,N] = size(Features);

%% Determine initial parameters

K = 30;

Beta1 = ones(length(Features(:,1)),1); %shrinking factor for features

trial_No = 10; % no of times to shuffle

%%

C = zeros(trial_No,1);
MSE = zeros(trial_No,1);

trial = 1;

%% Shuffle samples

Idx_New = randperm(N,N);
Features_New = zeros(size(Features));
Survival_New = zeros(size(Survival));
Censored_New = zeros(size(Censored));
for i = 1:N
Features_New(:,i) = Features(:,Idx_New(1,i));
Survival_New(:,i) = Survival(:,Idx_New(1,i));
Censored_New(:,i) = Censored(:,Idx_New(1,i));
end
Features = Features_New;
Survival = Survival_New;
Censored = Censored_New;

%% Assign samples to PROTOTYPE set, validation set (for model selection) ... 
%  and testing set (for model assessment):
%  The reason we call it "prototype set" rather than training set is 
%  because there is no training involved. Simply, the patients in the 
%  validation/testing set are matched to similar ones in the prototype
%  ("database") set.

K_cv = 3;
Folds = ceil([1:N] / (N/K_cv));

X_train = Features(:, Folds == 1);
X_valid = Features(:, Folds == 2);
X_test = Features(:, Folds == 3);

Survival_train = Survival(:, Folds == 1);
Survival_valid = Survival(:, Folds == 2);
Survival_test = Survival(:, Folds == 3);

Censored_train = Censored(:, Folds == 1);
Censored_valid = Censored(:, Folds == 2);
Censored_test = Censored(:, Folds == 3);

% Convert outcome from survival to alive/dead status using time indicator
t_min = min(Survival)-1;
t_max = max(Survival);
time = [t_min:1:t_max]';
Alive_train = TimeIndicator(Survival_train,Censored_train,t_min,t_max);
Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);

%%

% initialize
Y_test = nan(1,length(X_test(1,:)));
P_Center_Max = length(Y_test(1,:));

P_Center = 1;

% current point to label
Center = X_test(:,P_Center);

%% Compare to every other point using non-missing common features 
% initializa distance to each surrounding point
Dist = zeros(3,length(X_train(1,:)));
Dist(2,:) = Survival_train;
Dist(3,:) = Censored_train;
Dist(4,:) = 1:length(X_train(1,:)); %sample index

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

% sort by survival time
Dist = sortrows(Dist,2);

% predict survival based on KM
[t,f,~,~] = KM(Dist(:,2), Dist(:,3));
pAUC = sum(diff(t) .* f(1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
Yhat1 = pAUC * max(t); %since f is maximally 1, max possible survival is tmax*1

%get alive/dead status of K nearest neighbours
Alive_surround = Alive_train(:,Dist(:,4));

%% plot KM curve for nearest neighbours
KMPlot(Dist(:,2), Dist(:,3), ones(K,1), 'a'); 
hold on;

% plot predicted survival time
[Yhat1,~] = meshgrid(Yhat1,1:100);
plot(Yhat1,1./[1:100], 'b--');
hold off;

%% plot illustration of time indicator

% separate color channels
Alive_surround(Alive_surround == 0) = 0.5;
Alive_surround(isnan(Alive_surround) ==1) = 0;
Alive_only = Alive_surround;
Alive_only(Alive_only == 0.5) = 0;
Dead_only = Alive_surround;
Dead_only(Dead_only == 1) = 0;
%reducing glare
Alive_only = 0.75 .* Alive_only; 
Dead_only = 0.75 .* Dead_only; 
% plotting
im(:,:,1) = Dead_only';
im(:,:,2) = Alive_only';
im(:,:,3) = zeros(size(Alive_surround'));
figure(2)
image(im)
hold on
xlim([0,max(Dist(:,2))]);

%% Predict survival of current patient

% return original notation
Alive_surround(Alive_surround==0) = nan;
Alive_surround(Alive_surround==0.5) = 0;

UnknownStatus = isnan(Alive_surround);
Alive_surround(isnan(Alive_surround)==1) = 0;
% average alive/dead status at each time point (ignoring unknown status)
Alive_center = sum(Alive_surround, 2) ./ (K - sum(UnknownStatus, 2));

% Predict survival based on timeIndicator
Yhat2 = sum(Alive_center);

% plot predicted survival time
[Yhat2,~] = meshgrid(Yhat2,1:100);
plot(Yhat2,100./[1:100], 'b--');
hold off;

% plot KM-like curve for central patient
figure(3)
plot([1:length(Alive_center)]',Alive_center); 
hold on;
xlim([0,max(Dist(:,2))]);
plot(Yhat2,1./[1:100], 'b--');
hold off;

%% plot predicted illustrated survival for patient

% separate color channels
Dead_center = Alive_center;
Alive_center(Alive_center<0.5) = 0;
Dead_center(Dead_center>=0.5) = 0;
% 1 minus Dead_Center (so that redder is "more dead")
Dead_center(Alive_center==0) = 1 - Dead_center(Alive_center==0);
%reducing glare
Alive_center = 0.75 .* Alive_center; 
Dead_center = 0.75 .* Dead_center; 
% plotting
im2(:,:,1) = Dead_center'; 
im2(:,:,2) = Alive_center; 
im2(:,:,3) = zeros(size(Alive_center'));
figure(4)
image(im2)
hold on;
xlim([0,max(Dist(:,2))]);

% plot predicted survival time
[Yhat2,~] = meshgrid(Yhat2,1:100);
plot(Yhat2,2./[1:100], 'b--');
hold off;