% based on bayesLinRegDemo2d

%% Bayesian infernece for simple linear regression with known noise variance
% The goal is to reproduce fig 3.7 from Bishop's book
% We fit the linear model f(x,w) = w0 + w1 x and plot the posterior over w.
%%

function thesisbayesLinReg()
setSeed(1); % seed 0 reproduces Bishop's figure
a0 = -0.3; %Parameters of the actual underlying model that we wish to recover
a1 = 0.5;  %We will estimate these values with w0 and w1 respectively. 

trainingPoints = 20;    % Number of (x,y) training points
noiseSD = 0.2;          % Standard deviation of Gaussian noise, applied to actual underlying model. 
priorPrecision = 1.0;   % Fix the prior precision, alpha. We will use a zero-mean isotropic Gaussian.
likelihoodSD = noiseSD; % Assume the likelihood precision, beta, is known.
likelihoodPrecision = 1/(likelihoodSD)^2; 

%% Generate the training points
xtrain = -1 + 2*rand(trainingPoints,1);
model = struct('mu', 0, 'Sigma', noiseSD);
noise = gaussSample(model, trainingPoints);
ytrain = a0 + a1*xtrain + noise;

%% Plot
% Number of successive data points for which likelihood distributions will
% be graphed. The prior and the last data point are always plotted so 0 <=
% iter <= trainingPoints - 1.
nrows = 2; ncols = 2;

% Plot the prior distribution over w0, w1
subplot2(nrows,ncols,1,1);

priorMean = [0;0];
priorSigma = eye(2)./priorPrecision; %Covariance Matrix
priorPDF = @(W)gaussProb(W,priorMean',priorSigma);
contourPlot(priorPDF,[]);

% Plot the likelihood
subplot2(nrows,ncols,1,2);
likfunc = @(w) likelihood(w,xtrain,ytrain,likelihoodSD);
contourPlot(likfunc,[a0,a1]);

% Plot the posterior over all of the training data. 
subplot2(nrows,ncols,2,1);
[postW,mu,sigma] = update([ones(trainingPoints,1),xtrain],ytrain,likelihoodPrecision,priorMean,priorSigma);
contourPlot(postW,[a0,a1]);
disp(mu)


% Plot sample lines whose parameters are drawn from the posterior. 
subplot2(nrows,ncols,2,2);
xsorted = linspace(-1,1,20)';
x = [ones(20,1),xsorted];
fmu = x*mu;
fvar = diag(x*sigma*x' + 0.2^2); % noisy prediction
%plot(xsorted,fmu,'xk');
%plot(xsorted,fmu-2*sqrt(fvar),'xk');
%plot(xsorted,fmu+2*sqrt(fvar),'xk');
plotMeanAndStd(xsorted,fmu,2*sqrt(fvar),[7 7 7]/9);
plotSampleLines(mu',sigma,5,[xtrain,ytrain]);

% Add titles
subplot2(nrows,ncols,1,1);
set(gca,'FontSize',18);
title('prior');

subplot2(nrows,ncols,1,2);
set(gca,'FontSize',18);
title('log likelihood');

subplot2(nrows,ncols,2,1);
set(gca,'FontSize',18);
title('posterior');

subplot2(nrows,ncols,2,2);
set(gca,'FontSize',18);
title('predictive distribution');

saveas(gcf,'/home/trung/Dropbox/phdDocuments/trungthesis/figs/regression1d.eps','epsc');
end

function lik = likelihood(w,x,y,likelihoodSD)
n = size(x,1);
lik = uniGaussPdf(y,[ones(n,1),x]*w',(likelihoodSD.^2)*ones(n,1));
lik = exp(sum(log(lik)));
% or log?
% lik = sum(log(lik));
end

function plotSampleLines(mu, sigma, numberOfLines,dataPoints)
% Plot the specified number of lines of the form y = w0 + w1*x in [-1,1]x[-1,1] by
% drawing w0, w1 from a bivariate normal distribution with specified values
% for mu = mean and sigma = covariance Matrix. Also plot the data points as
% blue circles. 
for i = 1:numberOfLines
    model = struct('mu', mu, 'Sigma', sigma);
    w = gaussSample(model);
    func = @(x) w(1) + w(2)*x;
    fplot(func,[-1,1,-1,1],'r');
    hold on;
end
axis square;
set(gca,'XTick',[-1,0,1]);
set(gca,'YTick',[-1,0,1]);
xlabel(' x ', 'FontSize',16);
ylabel(' y ','Rotation',0,'FontSize',16);
if(size(dataPoints,2) == 2)
    hold on;
    plot(dataPoints(:,1),dataPoints(:,2),'xb');    
end

%Generates a colour filled contour plot of the bivariate function, 'func'
%over the domain [-1,1]x[-1,1], plotting it to the current figure. Also plots
%the specified point as a white cross. 
end
function contourPlot(func,trueValue)
stepSize = 0.05; 
[x,y] = meshgrid(-1:stepSize:1,-1:stepSize:1); % Create grid.
[r,c]=size(x);

data = [x(:) y(:)];
p = zeros(size(data,1),1);
for i=1:numel(p)
  p(i) = func(data(i,:));
end
p = reshape(p, r, c);
 
contourf(x,y,p,256,'LineColor','none');
colormap(jet(256));
axis square;
set(gca,'XTick',[-1,0,1]);
set(gca,'YTick',[-1,0,1]);
xlabel(' w_0 ','FontSize',16);
ylabel(' w_1 ','Rotation',0,'FontSize',16);
if(length(trueValue) == 2)
    hold on;
    plot(trueValue(1),trueValue(2),'+w');
end
end
%
% Given the mean = priorMu and covarianceMatrix = priorSigma of a prior
% Gaussian distribution over regression parameters; observed data, xtrain
% and ytrain; and the likelihood precision, generate the posterior
% distribution, postW via Bayesian updating and return the updated values
% for mu and sigma. xtrain is a design matrix whose first column is the all
% ones vector.
function [postW,postMu,postSigma] = update(xtrain,ytrain,likelihoodPrecision,priorMu,priorSigma)

postSigma  = inv(inv(priorSigma) + likelihoodPrecision*xtrain'*xtrain); 
postMu = postSigma*inv(priorSigma)*priorMu + likelihoodPrecision*postSigma*xtrain'*ytrain; 
postW = @(W)gaussProb(W,postMu',postSigma);

end
