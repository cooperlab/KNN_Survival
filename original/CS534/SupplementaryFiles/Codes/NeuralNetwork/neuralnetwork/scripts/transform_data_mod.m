function [newdata] = transform_data_mod(data,name)

% convert time to years
survival = round(data.Survival/365)+1;
features = data.Features;

censored  = logical(data.Censored);
dead      = nan(sum(survival),1);
tfeatures = nan(sum(survival),size(features,1));



% loop through features to create new matrix
counter = 0;
counter2 = 0
dead = []
tf = []
sfeat = []
subs = []
cens = []
for i = 1:size(features,2)
  if censored(i)
    tf = [tf repmat(features(:,i),1,survival(i))];
    dead = [dead repmat(0,1,survival(i))];
    sfeat = [sfeat 1:survival(i)];
    subs = [subs repmat(i,1,survival(i))];
    cens = [cens repmat(censored(i),1,survival(i))];
  else
    tf = [tf repmat(features(:,i),1,2*survival(i))];
    dead = [dead repmat(0,1,survival(i))];
    dead = [dead repmat(1,1,survival(i))];

    sfeat = [sfeat 1:2*survival(i)];
    subs = [subs repmat(i,1,2*survival(i))];
    cens = [cens repmat(censored(i),1,2*survival(i))];
  end
end

  
% store data params
newdata.features = zscore([tf; sfeat]');
newdata.survival = [dead; ~dead]';
newdata.survival2 = [~dead]';
newdata.subs = subs';
newdata.censored = cens';
for i = 1:5
  B = randperm(size(features,2));
  tr(i,:) = ismember(newdata.subs,B(1:round(0.8*length(B))));
  te(i,:) = ~ismember(newdata.subs,B(1:round(0.8*length(B))));
end



newdata.tr = tr;
newdata.te = te;

save(sprintf('~/machinelearningproject/neuralnetwork/%s',name),'newdata')






