function Alive = TimeIndicator(Survival,Censored, t_min, t_max)

%
% This converts survival-censored data into alive-dead data by adding
% a time indicator variable.
% samples on columns

%% Sample Inputs
% clear ; close all ; clc ; 
% 
% N = 100; % sample size
% p = 12; %no of features
% 
% %
% % FOR THE FOLLOWING: features in rows, samples on columns
% %
% Survival = randi([1,300],1,N); % survival of training sample
% Censored = randi([0,1],1,N); % censorship of training sample: 1=alive
% t_min = 0;
% t_max = max(Survival);

%%

N = length(Survival);

% Generate sample index
Idx = 1:length(Survival);

% Generate survival time range
Surv_min = min(Survival);
Surv_max = max(Survival);
time = [t_min:1:t_max]';

% Initialize alive/dead indicator (1=alive, 0=dead) over entire time range
[Alive,~] = meshgrid(ones(size(Idx)),time);

% Loop through samples and keep correct initialized indicator
for i = 1:N

S = Survival(1,i);

if Censored(1,i) == 0
    Alive(S:end,i) = 0;
elseif Censored(1,i) == 1
    Alive(S:end,i) = nan; % cannot ascertain alive/dead status
end

end

end