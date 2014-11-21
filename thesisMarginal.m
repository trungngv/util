%% marginal decomposition
rng(1110, 'twister');
N = 20;
D = 1;
x = linspace(-7,7,N)';
hyp0 = [log(1); log(1)]; % [lengthscale, sigma]
f = sample_gp(x,@covSEard,hyp0,1);
sigma = 0.1;
y = f + sigma*randn(N,1);

ls = 0.1:0.1:4; % all lengthscales
ns = numel(ls);
datafit = zeros(ns,1);
npenalty = datafit;
marginal = datafit;
for i=1:numel(ls)
  hyp = [log(ls(i)); log(1)];
  Ky = covSEard(hyp,x) + sigma^2*eye(N);
  datafit(i) = -0.5*y'*(Ky\y);
  npenalty(i) = -0.5*log(det(Ky));
  marginal(i) = datafit(i) + npenalty(i) - 0.5*N*log(2*pi);
end
figure; hold on;
fs = 20; % fontsize
plot(ls, datafit,'g--','LineWidth',2);
plot(ls, npenalty,'r-.','LineWidth',2);
plot(ls, marginal,'b','LineWidth',2);
ylim([-100,50]);
xlabel('characteristic lengthscale','FontSize',fs);
ylabel('log probability','FontSize',fs);
set(gca,'FontSize',fs);
legend('data fit', 'neg. complexity', 'marginal likelihood', 'Location','Best');
legend boxoff
saveas(gcf,'/home/trung/Dropbox/phdDocuments/trungthesis/figs/marginal.eps','epsc');

%% local minima of marginal to show contour
%gprDemoMarglik

% ls = linspace(0.1,80,91);
% sigma = linspace(0.01,3,91);
% [x1 x2] = meshgrid(ls,sigma);
% Z = [];
% for i=1:numel(x1)
%   hyp = [log(x1(i)); log(1)];
%   Ky = covSEard(hyp,x) + x2(i)^2*eye(N);
%   Z(i) = -0.5*y'*(Ky\y) - 0.5*log(det(Ky)) - 0.5*N*log(2*pi);
% end
% Z(Z < -200) = -200;
% minZ = min(Z); maxZ = max(Z);
% Z = reshape(Z,numel(sigma),numel(ls));
% 
% figure; hold on;
% [cs,hh]=contour(x1,x2,Z,-[18 20 23 26 29 30.2 30.6 31 37 45]);
% colorbar;
% 
% % find a minimum
% hyp.cov = [log(1); log(1)]; hyp.lik = log(0.5);
% [hyp, fmin] = minimize(hyp,@gp,-100,@infExact,[],@covSEard,@likGauss,x,y);
% plot(exp(hyp.cov(1)),exp(hyp.lik),'+','MarkerSize',14,'LineWidth',2)
% 
% % find another minimum
% hyp.cov = [log(2); log(1)]; hyp.lik = log(1);
% [hyp, fmin] = minimize(hyp,@gp,-100,@infExact,[],@covSEard,@likGauss,x,y);
% plot(exp(hyp.cov(1)),exp(hyp.lik),'o','MarkerSize',14,'LineWidth',2)
% xlim([0,40]);
% 
% set(hh,'LineWidth',2)
% set(gca,'XScale','log','YScale','log')
% xlabel('characteristic lengthscale')
% ylabel('noise standard deviation')
