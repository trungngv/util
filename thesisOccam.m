clear; close all
x = -5:0.01:5;
simple = normpdf(x,0,0.3);
justright = normpdf(x,0,0.8);
toocomplex = normpdf(x,0,2);
ylocation = 0.8;
figure; hold on;
plot(x,simple,'b-.');
plot(x,justright,'g');
plot(x,toocomplex,'r--');
line([ylocation,ylocation],[0,normpdf(ylocation,0,0.8)],'Color','k','LineStyle',':','LineWidth',2);
set(gca,'XTick',[ylocation],'FontSize',18);
set(gca,'YTick',[],'FontSize',18);
set(gca,'XTickLabel',{'y'},'FontSize',18, 'FontWeight','bold')
set(gca,'FontWeight','normal');
xlabel('all possible datasets');
ylabel('marginal likelihood, p(D|m)');
legend({'simple model', 'appropriate model', 'complex model'},'Location','Best');
legend boxoff
set(gca,'FontSize',20);
saveas(gcf, '/home/trung/Dropbox/phdDocuments/trungthesis/figs/occam.eps','epsc');
