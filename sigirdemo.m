function sigirdemo()
%% linear reg
N = 7;
[xtrain, ytrain, xtest, ytestNoisefree, ytest] = ...
    polyDataMake('sampling','thibaux', 'n', N);
deg = 1;

pp = preprocessorCreate('rescaleX', true, 'poly', deg, 'addOnes', true);
model = linregFit(xtrain, ytrain, 'preproc', pp);
ypredTrain = linregPredict(model, xtrain);
ypredTest = linregPredict(model, xtest);

figure; hold on;
plot(xtrain,ytrain,'.b', 'markersize', 30);
plot(xtest, ypredTest, 'k', 'linewidth', 3, 'markersize', 20);
xlabel('input, x','FontSize',25)
ylabel('f(x)','FontSize',25)
set(gca,'FontSize',25);
saveas(gcf,'/home/trung/Dropbox/phdDocuments/slides-sigir14/linreg.eps','epsc');
saveas(gcf,'/home/trung/Dropbox/phdDocuments/slides-sigir14/linreg.pdf');

%% bayesian linear regression
model = linregFitBayes(xtrain, ytrain, 'prior', 'EB');
[mu, sig2] = linregPredictBayes(model, xtest);
figure; hold on;
plotMeanAndStd(xtest,mu,2*sqrt(sig2),[7 7 7]/9);
plot(xtrain,ytrain,'.b', 'markersize', 30);
xlabel('input, x','FontSize',25)
ylabel('f(x)','FontSize',25)
set(gca,'FontSize',25);
saveas(gcf,'/home/trung/Dropbox/phdDocuments/slides-sigir14/bayeslinreg.eps','epsc');
saveas(gcf,'/home/trung/Dropbox/phdDocuments/slides-sigir14/bayeslinreg.pdf');

end

function [postW,postMu,postSigma] = update(xtrain,ytrain,likelihoodPrecision,priorMu,priorSigma)

postSigma  = inv(inv(priorSigma) + likelihoodPrecision*xtrain'*xtrain); 
postMu = postSigma*inv(priorSigma)*priorMu + likelihoodPrecision*postSigma*xtrain'*ytrain; 
postW = @(W)gaussProb(W,postMu',postSigma);

end

function forsigir()
rng(1110,'twister');
% some function
covfunc = @covSEard;
covhyp = [0.5;1];
x = (-5:0.1:5)';
y = sample_gp(x,covfunc,covhyp,1);
y = y + 0.5*randn(size(x));

% only few observations
indice = [5; 12; 20; 45; 70; 95];
x0 = x(indice); y0 = y(indice);

% the data & hypotheses
figure; hold on;
plot(x0,y0,'ok','MarkerSize',14,'MarkerFaceColor','k');
line1 = line([-5,5],[-4,1]);
line2 = line([-5,5],[-1.5,-1.5]);
line3 = line([-5,5],-1.9-0.25*[-5,5]); % linear fit
set(line1,'LineWidth',2,'color','r');
set(line2,'LineWidth',2,'color','g');
set(line3,'LineWidth',2,'color','b');
xlabel('x','FontSize',24);
ylabel('f(x)','FontSize',24);
set(gca,'FontSize',20);
saveas(gcf,'demo1.pdf','pdf');
saveas(gcf,'demo1.eps','epsc');
saveas(gcf,'demo1.png','png');

% 
figure; hold on;
%plot(x0,y0,'+k','MarkerSize',14);
plot(x0,y0,'ok','MarkerSize',14,'MarkerFaceColor','k');
line3 = line([-5,5],-1.9-0.25*[-5,5]); % linear fit
set(line3,'LineWidth',2,'color','k');
xlabel('x','FontSize',24);
ylabel('f(x)','FontSize',24);
set(gca,'FontSize',20);
saveas(gcf,'demo2.pdf','pdf');
saveas(gcf,'demo2.eps','epsc');
saveas(gcf,'demo2.png','png');

% gp prior
fs = sample_gp(x,covfunc,covhyp,5);
K = feval(covfunc, [2,1.5], x);
figure; hold on;
plotMeanAndStd(x,zeros(size(x)),sqrt(diag(K)),[7 7 7]/8);
%plot(x0,y0,'+k','MarkerSize',14);
plot(x0,y0,'ok','MarkerSize',14,'MarkerFaceColor','k');
plot(x,fs(:,1),'-r','LineWidth',2);
%plot(x,fs(:,2),'-g','LineWidth',2);
plot(x,fs(:,3),'-g','LineWidth',2);
%plot(x,fs(:,4),'-b','LineWidth',2);
plot(x,fs(:,5),'-b','LineWidth',2);
xlabel('x','FontSize',24);
ylabel('f(x)','FontSize',24);
set(gca,'FontSize',20);
saveas(gcf,'demo3.pdf','pdf');
saveas(gcf,'demo4.png','png');
saveas(gcf,'demo3.eps','epsc');

% gp posterior
figure; hold on;
rng(4,'twister');
hyp.cov = [rand;rand]; hyp.lik = log(0.5);
[hyp nlm] = minimize(hyp,@gp,-1000,@infExact,[],covfunc,@likGauss,x0,y0);
sn2 = exp(2*hyp.lik);

% predictive posterior (noiseless) and samples
[ymu,yvar,fmu,fvar] = gp(hyp,@infExact,[],covfunc,@likGauss,x0,y0,x);
plotMeanAndStd(x,fmu,2*sqrt(fvar),[7 7 7]/8);

[ymu,yvar,fmu,fvar] = gp(hyp,@infExact,[],covfunc,@likGauss,x0,y0,x);
K = feval(covfunc, hyp.cov, x0);
Ks = feval(covfunc, hyp.cov, x, x0); % K(x,x0)
Kss = feval(covfunc, hyp.cov,x);
Sigma = Kss - Ks*((K+sn2*eye(size(K)))\Ks');
nsamples = 5;
L = jit_chol(Sigma)';
U = randn(size(x,1),nsamples);
fs = repmat(fmu,1,nsamples)+L*U;
%fs = mvnrnd(fmu',Sigma,2)';
plot(x,fs(:,2),'-r','LineWidth',2);
plot(x,fs(:,3),'-g','LineWidth',2);
plot(x,fs(:,5),'-b','LineWidth',2);
%plot(x0,y0,'+k','MarkerSize',14);
plot(x0,y0,'ok','MarkerSize',14,'MarkerFaceColor','k');
xlabel('x','FontSize',24);
ylabel('f(x)','FontSize',24);
set(gca,'FontSize',20);
saveas(gcf,'demo4.pdf','pdf');
saveas(gcf,'demo4.png','png');
saveas(gcf,'demo4.eps','epsc');

% posterior
% [~,~,~,~,~,post] = gp(hyp,@infExact,[],covfunc,@likGauss,x0,y0,1);
% K = feval(covfunc, hyp.cov, x0);
% postmean = K*post.alpha;
% postCov = inv(inv(K)+diag(post.sW.^2));
% plotMeanAndStd(x0,postmean,2*sqrt(diag(postCov)),[7 7 7]/8);
%plot(x0,y0,'+r','MarkerSize',14);
end
