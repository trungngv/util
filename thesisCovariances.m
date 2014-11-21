% plot covariance functions and GP samples
rng(1110,'twister');
x = linspace(0,10,100)';
covfunc = @covSEard;
length = [1; 2; 5]; % l^2
styles = {'r-','g--','b-.'};
fs = 20;

% plot kernels
figure; hold on;
for i=1:3
  k = exp(-(x.^2)/length(i)^2);
  plot(x, k, styles{i},'LineWidth',2);
end
legend('l = 1', 'l = 2', 'l = 5','Location','Best');
legend boxoff
xlabel('input distance, r','FontSize',fs);
ylabel('covariance, \kappa(r)','FontSize',fs);
set(gca,'FontSize',fs);
saveas(gcf,'/home/trung/Dropbox/phdDocuments/trungthesis/figs/covSE.eps','epsc');

% plot samples
figure; hold on;
for i=1:3
  f = sample_gp(x,covfunc,log([length(i);2]),1);
  plot(x, f, styles{i},'LineWidth',2);
end
%legend('l = 1', 'l = 2', 'l = 5');
xlabel('input, x', 'FontSize',fs);
ylabel('output, f(x)','FontSize',fs);
set(gca,'FontSize',fs);
saveas(gcf,'/home/trung/Dropbox/phdDocuments/trungthesis/figs/covSEsamples.eps','epsc');
