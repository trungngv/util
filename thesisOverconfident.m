%% Posterior predictive density for Bayesian linear Regression in 1d with Polynomial Basis 
% We use a gaussian prior with fixed noise variance
% We plot the posterior predictive density, and samples from it
%%

% This file is from pmtk3.googlecode.com

%setSeed(1);
%xtrain = (-10:0.1:10)';
%xtest = (-10:0.11:10)';
func = @(x) x.^2 - x + 1;
xtrain = linspace(-2,4,10)';
ytrain = func(xtrain);
sigma2 = 0.5;
ytrain = ytrain + sqrt(sigma2)*randn(size(ytrain,1),1);
xtest = linspace(-2,4,100)';
ytest = func(xtest);

%[xtrain, ytrain, xtest, ytestNoisefree, ytest, sigma2] = ...
%  polyDataMake('sampling', 'sparse', 'deg', 2);
deg = 1;
addOnes = false;
Xtrain = degexpand(xtrain, deg, addOnes);
Xtest  = degexpand(xtest, deg, addOnes);
fs = 18;

%% MLE
modelMLE = linregFit(Xtrain, ytrain); 
[mu, v] = linregPredict(modelMLE, Xtest);

fig1=figure;
hold on;
plot(xtest, mu,  'k-', 'linewidth', 3, 'displayname', 'prediction');
%plot(xtest, ytestNoisefree,  'b:', 'linewidth', 3, 'displayname', 'truth');
plot(xtrain,ytrain,'rx','markersize', 14, 'linewidth', 3, ...
     'displayname', 'training data');
NN = length(xtest);
ndx = 1:5:NN; % plot subset of errorbars to reduce clutter
sigma = sqrt(v);
h=legend('location', 'northwest');
set(h, 'fontsize', fs); 
errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx));
title('plugin approximation (MLE)', 'fontsize', fs);
close 
%printPmtkFigure('linregPostPredPlugin')


%% Bayes
%[model logev] = linregFitBayes(Xtrain, ytrain, ...
%  'prior', 'gauss', 'alpha', 0.001, 'beta', 1/sigma2);
[model logev] = linregFitBayes(Xtrain, ytrain, 'prior', 'gauss', 'alpha', 0.001, 'beta', 1/sigma2);
fprintf('degree = %d, logev = %.2f\n', deg, logev)
[mu, v] = linregPredictBayes(model, Xtest);
fig2=figure;
hold on;
plot(xtest, mu,  'k-', 'linewidth', 3, 'displayname', 'prediction');
%plot(xtest, ytestNoisefree,  'b:', 'linewidth', 3, 'displayname', 'truth');
plot(xtrain,ytrain, 'bx', 'markersize', 12, 'linewidth', 3, ...
    'displayname', 'training data');
NN = length(xtest);
ndx = 1:5:NN; % plot subset of errorbars to reduce clutter
sigma = sqrt(v);
%h=legend('location', 'northwest');
%set(h,'fontsize', fs); 
errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx),'k');
xlim([-2.1,4]);
set(gca,'Fontsize',fs);
title('Posterior predictive distribution', 'fontsize', fs);
saveas(gcf, '/home/trung/Dropbox/phdDocuments/trungthesis/figs/overconfident2.eps','epsc');

%% Plot samples from posterior predictive
S = 5;
%model.wN = model.wMu; model.VN = model.wCov;
ws = gaussSample(struct('mu', model.wN, 'Sigma', model.VN), S);
figure;
hold on;
plot(xtrain,ytrain, 'bx', 'markersize', 12, 'linewidth', 3, ...
    'displayname', 'training data');
for s=1:S
  tmp = modelMLE;  tmp.w = ws(s,:)';
  [mu] = linregPredict(tmp, Xtest);
  plot(xtest, mu, 'k-', 'linewidth', 2);
end
xlim([-2.1,4]);
set(gca,'Fontsize',fs);
title('Functions sampled from the posterior', 'fontsize', fs)
saveas(gcf, '/home/trung/Dropbox/phdDocuments/trungthesis/figs/overconfident1.eps','epsc');

