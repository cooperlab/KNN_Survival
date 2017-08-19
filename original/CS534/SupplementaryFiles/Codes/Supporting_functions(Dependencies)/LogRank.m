function [p ChiSq] = LogRank(Survival, Censored, Labels)
%Implements the logrank test to compare survival between multiple groups.
%Follows "The logrank test" in Statistics Notes, by Bland and Altman, 
%BMJ 328 : 1073 doi: 10.1136/bmj.328.7447.1073 (Published 29 April 2004).
%inputs:
%Survival - N-length double vector indicating survival times.  
%Censored - N-length logical vector indicating right-censoring status.
%Labels - N-length double vector with discrete values indicating patient grouping.
%outputs:
%P - scalar significance of logrank test.
%ChiSq - scalar chi-square statistic.

%identify unique groups
Groups = unique(Labels);

%initialize survival, censoring containers
CellSurvival = cell(1,length(Groups));
CellCensored = cell(1,length(Groups));

%cell-i-fy survival, censoring by group for easy access
for i = 1:length(Groups)
    CellSurvival{i} = Survival(Labels == Groups(i));
    CellCensored{i} = logical(Censored(Labels == Groups(i)));
end

%get unique event times
Times = unique(Survival);

%initialize container for expected survival times
Expected = zeros(1,length(Groups));

%calculate expected survival times
for i = 1:length(Times)
    
    %find living and uncensored in pooled groups
    Living = sum(Survival >= Times(i));
    
    %find deceased in each group, censoring not related to prognosis
    Events = zeros(1,length(Groups));
    for j = 1:length(Groups)
        Events(j) = sum(CellSurvival{j}(~CellCensored{j}) == Times(i));
    end
    
    %calculate risks
    Risk = sum(Events) / Living;
    
    %calculate expected number of deaths
    for j = 1:length(Groups)
        Expected(j) = Expected(j) + sum(CellSurvival{j} >= Times(i)) * Risk;
    end

end

%generate chi-square statistic
ChiSq = 0;
for j = 1:length(Groups)
    ChiSq = ChiSq + (sum(~CellCensored{j}) - Expected(j))^2 / Expected(j);
end

%calculate p-value
p = 1-chi2cdf(ChiSq, length(Groups)-1);

end

% function test()
% 
% Survival1 = [6 13 21 30 31 37 38 47 49 50 63 79 80 82 82 86 98 149 202 219];
% Censored1 = logical([0 0 0 0 1 0 0 1 0 0 0 0 1 1 1 0 0 1 0 0]);
% Survival2 = [10 10 12 13 14 15 16 17 18 20 24 24 25 28 30 33 34 35 37 40 40 40 46 48 70 76 81 82 91 112 181];
% Censored2 = logical([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0]);
% 
% Survival = [Survival1 Survival2];
% Censored = [Censored1 Censored2];
% Labels = [1*ones(1,length(Survival1)) 2*ones(1,length(Survival2))];
% 
% [p ChiSq] = LogRank(Survival, Censored, Labels
% 
% %p should be < 0.01, ChiSq = 6.88
% 
% end