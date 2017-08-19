%
% Using Cox proportional hazards regression with elastic net to predict
% survival
%

clear ; close all ; clc ; 

%% Load data
load('/home/mohamed/Desktop/Research_Other/GynecologicRMS/Data/RMSData_OS.mat')

Data = RMSData.Predictors;
Survival = RMSData.Survival;
Censored = RMSData.Censored;
Patients = RMSData.PatientId;

%% Putting testing sample aside

N_all = length(Data(:,1));
N_test = ceil(0.2*N_all); %testing sample size

% indices of testing samples
Idx_test = (randperm(N_all))'; %random assignment
Idx_test = Idx_test(1:N_test);

Data1 = Data;
Survival1 = Survival;
Censored1 = Censored;
Patients1 = Patients;
j = 0;
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

fit = coxphfit(X,Survival,'Censoring',Censored);

%% Calculating Concordance index for chosen model (evaluated on testing sample)

% training concordance (reverse training error)
c_train = cIndex(fit, Data, Survival, Censored)

% testing concordance (reverse testing error)
c_test = cIndex(fit, Data_test, Survival_test, Censored_test)