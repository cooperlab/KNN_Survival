function h = KMPlot(Survival, RightCensored, Labels, Names)

%Generates survival plot for labeled samples and returns plot handle.
%inputs:
%Survival - N-length vector of positive outcome values.
%RightCensored - N-length vector of {0, 1} indicating right-censored status.
%Labels - N-length vector of sample types.
%Names - unique(Labels) length cell array of strings of type names.
%outputs:
%h - figure handle.

%plot colors
colors = {'r', 'g', 'b', 'k', 'c', 'm', 'y'};

%figure handle
h = figure;

%generate group KM plot
for i = 1:max(Labels)
    
    %calculate empircal cdf
    [x f cx{i} cf{i}] = KM(Survival(Labels == i),...
                            RightCensored(Labels == i));
    
    %stair plot for class 'i'
    stairs(x, f, colors{mod(i-1,length(colors))+1}); hold on;
    
end

%format KM plot
xlabel('Time'); ylabel('Outcome'); legend(Names);

%plot censored points
for i = 1:max(Labels)
    if(~isempty(cx{i}))
        plot(cx{i}, cf{i}, 'k+');
    end
end

%set axes limits
ylim([0 1]);
xlim([0 max(Survival)]);

end