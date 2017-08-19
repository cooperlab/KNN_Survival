function [x f cx cf] = KM(Survival, RightCensored)
%Generates traces for KM plot with 'steps'.
%inputs:
%Survival - N-length vector of positive outcome values.
%RightCensored - N-length vector of {0, 1} indicating right-censored status.
%outputs:
%x - times (abcissa).
%f - corresponding frequencies.
%cx - censored times (abcissa).
%cf - corresponding censored frequencies.

%find unique times
x = unique(Survival(~RightCensored));

%initialize count vectors
f = zeros(size(x));
d = f; n = d;

%generate counts
for i = 1:length(x)
    n(i) = sum(Survival >= x(i));
    d(i) = sum(Survival(~RightCensored) == x(i));
end

%calculate KM plot
f = (n - d) ./ n;
f = cumprod(f);

%correct origin
if(size(x,1) > size(x,2))
    x = [0; x];
    f = [1; f];
else
    x = [0 x];
    f = [1 f];
end

%initialize outputs 'cx', 'cf'
cx = Survival(RightCensored > 0);
cf = zeros(1,sum(RightCensored > 0));

%find location for censoring points
for i = 1:length(cx)
    
    %find first 'x' greater than 'cx(i)'
    idx = find(x < cx(i));
    
    %record corresponding 'f'
    cf(i) = f(idx(end));
    
end

%check for end condition (if last sample is censored)
[~, idx] = max(Survival);
if(RightCensored(idx))
    if(size(x,1) > size(x,2))
        x = [x; max(Survival)];
        f = [f; f(end)];
    else
        x = [x max(Survival)];
        f = [f f(end)];
    end
end

end