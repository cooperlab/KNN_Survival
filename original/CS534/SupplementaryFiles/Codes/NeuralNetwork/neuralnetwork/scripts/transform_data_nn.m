function [newdata] = transform_data_nn(data,name)

% convert signal to 
survival = round(data.Survival/365)+1;
features = data.Features;

censored = logical(data.Censored);
dead    = nan(sum(survival),1);
tfeatures = nan(sum(survival),size(features,1));



% loop through features to create new matrix
counter = 0;
counter2 = 0
for i = 1:size(features,2)
  for j = 1:survival(i)
    counter = counter + 1;
    featnum(counter) = i;
    sfeat(counter) = j;
    tfeatures(counter,:) = features(:,i)';
    
    
    % censored and survived
    if censored(i) & j==survival(i)
      dead(counter) = 0;
    % all survived times
   
    elseif j ~= survival(i)
      dead(counter) = 0;
    % dead
    else
      dead(counter) = 1;
    end
  
  
  end
end



newdata.features = [tfeatures sfeat'];

newdata.survival = [~dead];

save(sprintf('~/machinelearningproject/neuralnetwork/%s',name),'newdata')



