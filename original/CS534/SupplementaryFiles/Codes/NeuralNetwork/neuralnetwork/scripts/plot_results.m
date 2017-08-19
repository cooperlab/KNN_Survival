function plot_results(name)
% auc neural network
num_folds =5;
num_nodes =5;
nodes = {'1' '4' '7' '10' '13'}
figuresize(6,4,'inches')

% plot neural network results for original transformation
load(sprintf('~/machinelearningproject/neuralnetwork/%s_org_nn.mat',name))
counter = 0;
for i = 1:num_folds
  for j = 1:num_nodes
    counter = counter+1;
    [fpr,tpr,~,te_nn(i,j)] = perfcurve(double(y_test(i,:)), ...
				   double(yn_test(counter,:)),1);
  end
end
nanmean(te_nn)
nanstd(te_nn)
subplot(2,2,1)
boxplot(te_nn)
box off
set(gca,'XTickLabel',nodes)
xlabel('Number of Nodes')
ylabel('AUC')
title('ANN - Transformation 1')

% get c and auc for perceptron
load(sprintf('~/machinelearningproject/neuralnetwork/%s_org_p.mat',name))
for i = 1:num_folds
  g  = findgroups(group{i});
  c  = splitapply(@mean,cens{i}',g');
  yh = splitapply(@sum,y_hat_p{i},g');
  y  = splitapply(@sum,y_act{i},g');

  test(i) = cIndex2(yh+1,y,c);
  
  [fpr,tpr,~,auc(i)] = perfcurve(double(y_act{i}), ...
				 double(y_hat_p{i}),1);
end
nanmean(auc)
nanstd(auc)
subplot(2,2,2)
boxplot([auc; test]')
box off
set(gca,'XTickLabel',{'AUC','C-Index'})
xlabel('Metric')
ylabel('AUC or C-Index')
title('Perceptron - Transformation 1')


% plot neural network results for original transformation
load(sprintf('~/machinelearningproject/neuralnetwork/%s_mod_nn.mat',name))
counter = 0;
for i = 1:num_folds
  for j = 1:num_nodes
    counter = counter+1;
    [fpr,tpr,~,te_nn(i,j)] = perfcurve(double(y_test(i,:)), ...
				   double(yn_test(counter,:)),1);
  end
end
nanmean(te_nn)
nanstd(te_nn)
subplot(2,2,3)
boxplot(te_nn)
box off
set(gca,'XTickLabel',nodes)
xlabel('Number of Nodes')
ylabel('AUC')
title('ANN - Transformation 2')


% get c and auc for perceptron
load(sprintf('~/machinelearningproject/neuralnetwork/%s_mod_p.mat',name))
for i = 1:num_folds
  g  = findgroups(group{i});
  c  = splitapply(@mean,cens{i}',g');
  yh = splitapply(@sum,y_hat_p{i},g');
  y  = splitapply(@sum,y_act{i},g');

  test(i) = cIndex2(yh,y,c);
  
  [fpr,tpr,~,auc(i)] = perfcurve(double(y_act{i}), ...
				 double(y_hat_p{i}),1);
end
nanmean(auc)
nanstd(auc)
subplot(2,2,4)
boxplot([auc; test]')
box off
set(gca,'XTickLabel',{'AUC','C-Index'})
xlabel('Metric')
ylabel('AUC or C-Index')
title('Perceptron - Transformation 2')


saveas(gcf,sprintf('~/machinelearningproject/neuralnetwork/%s',name),'pdf')
keyboard

















