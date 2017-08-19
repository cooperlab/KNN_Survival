%
% Using Cox proportional hazards regression with elastic net to predict
% survival
%

clear ; close all ; clc ; 

%% Load data
%load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/ClinicalData.mat')
%load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/ProteomicData.mat')
%load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/GenomicData.mat')
load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/ClinicalData2.mat')

N_real = 10;

Perc_test = 0.2; %proportion allocated to testing sample

c_train = nan(N_real,1);
c_test = nan(N_real,1);

for k = 1:N_real

clear('Data','Survival','Censored','FeatureNames','Patients','Data_test','Survival_test','Censored_test','Patients_test','Beta');    

    
Data = ClinicalData.Data;
Survival = ClinicalData.Survival;
Censored = ClinicalData.Censored;
FeatureNames = ClinicalData.FeatureNames;
Patients = ClinicalData.Patients;
%------
% Data = ProteomicData.Data;
% Survival = ProteomicData.Survival;
% Censored = ProteomicData.Censored;
% FeatureNames = ProteomicData.FeatureNames;
% Patients = ProteomicData.Patients;
% %------
% Data = GenomicData.Data;
% Survival = GenomicData.Survival;
% Censored = GenomicData.Censored;
% FeatureNames = GenomicData.FeatureNames;
% Patients = GenomicData.Patients;

% reverse notation for censorship status (NOTE THAT THE GLMNET FUNCTION
% USES 1 FOR DEATH AND 0 FOR CENSORSHIP)
Censored = Censored+1;
Censored(Censored==2) = 0;

%% Putting testing sample aside

N_all = length(Data(:,1));
N_test = ceil(Perc_test*N_all); %testing sample size

% indices of testing samples
Idx_test = (randperm(N_all))'; %random assignment
Idx_test = Idx_test(1:N_test);

Data1 = Data;
Survival1 = Survival;
Censored1 = Censored;
Patients1 = Patients;
j = 0;

Data_test = nan(length(Idx_test(:,1)),length(Data(1,:)));
Survival_test = nan(length(Idx_test(:,1)),1);
Censored_test = nan(length(Idx_test(:,1)),1);

for i = 1:length(Idx_test(:,1))
    
    Idx_current = Idx_test(i,1);
    
    Data_test(i,:) = Data(i-j,:);
    Survival_test(i,:) = Survival(i-j,:);
    Censored_test(i,:) = Censored(i-j,:);
    Patients_test(i,:) = Patients(i-j,:);
    
    Data(i-j,:) = [];
    Survival(i-j,:) = [];
    Censored(i-j,:) = [];
    Patients(i-j,:) = [];
    
    j = j+1;
end

% saving test sample structure
TestingData.Data = Data_test;
TestingData.Survival = Survival_test;
TestingData.Censored = Censored_test;
TestingData.Patients = Patients_test;


%% Cox prop. hazards regression with Elastic Net

X = Data;
Y = [Survival,Censored];

% % Without cross-validation
% fit=glmnet(X,Y,'cox');
% glmnetPlot(fit);

% With cross-validation
K = 10; %how many folds?
cvfit=cvglmnet(X,Y,'cox',[],[],K);
%cvglmnetPlot(cvfit);

% Make predictions using model
% pred1 = glmnetPredict(fit,X); %without CV
% pred1 = cvglmnetPredict(cvfit,X); %with CV

%% Getting optimal model coefficients

% Extract index of optimum lambda
lambda_optimum = cvfit.lambda_1se; %maximum lambda within one SE of that which minimizes CV error
%lambda_optimum = cvfit.lambda_min;

% Extract index of optimal beta coefficients
Idx = (1:length(cvfit.lambda))';
Beta_Idx = cvfit.lambda - lambda_optimum;
Beta_Idx(Beta_Idx==0)=nan;
Beta_Idx(isnan(Beta_Idx)==0)=0;
Beta_Idx(isnan(Beta_Idx)==1)=1;
Beta_Idx = Beta_Idx .* Idx;
Beta_Idx = sum(Beta_Idx);

% extract optimal beta coefficients
Beta = cvfit.glmnet_fit.beta(:,Beta_Idx);

%% Calculating Concordance index for chosen model (evaluated on testing sample)

% training concordance (reverse training error)
c_train(k,1) = cIndex(Beta, Data, Survival, Censored);
% testing concordance (reverse testing error)
c_test(k,1) = cIndex(Beta, Data_test, Survival_test, Censored_test);

end

boxplot([c_train,c_test],'Labels',{'cIndex_train','cIndex_test'}); 
hold on ; 
title(['Training Vs Testing cIndex at ',num2str(1-Perc_test),'/',num2str(Perc_test),' train-test split']); 
ylabel('cIndex');