rng(1111,'twister');
% some function
covfunc = @covSEard;
covhyp = [1;1];
x0 = 15*rand(7,1)-7.5; % from -7.5 to 7.5
y0 = sample_gp(x0,covfunc,covhyp,1);
y0 = y0 + 0.1*randn(size(x0));
x = (-8:0.1:8)';

% the data & hypotheses
fs = 20;
figure; hold on;
hyp.cov = [log(3); log(1)];
hyp.lik = log(1);
hyp = minimize(hyp,@gp,-100,@infExact,[],covfunc,@likGauss,x0,y0);
% predictive posterior (noiseless) and samples
[ymu,yvar,fmu,fvar] = gp(hyp,@infExact,[],covfunc,@likGauss,x0,y0,x);
plotMeanAndStd(x,ymu,2*sqrt(yvar),[7 7 7]/8);
plot(x0,y0,'xb','MarkerSize',14,'LineWidth',2);
xlabel('input, x','FontSize',fs);
ylabel('output, y','FontSize',fs);
xlim([-8 8]);
set(gca,'FontSize',fs);
saveas(gcf,'/home/trung/Dropbox/phdDocuments/trungthesis/figs/gpr.eps','epsc');
