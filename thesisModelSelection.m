%% Bayesian model selection demo for polynomial regression
% This illustartes that if we have more data, Bayes picks a more complex model.
% We use variational Bayes (VB) to tune the hyper-parameter and approximate the marginal likelihood

% This file is from pmtk3.googlecode.com

% Based on a demo by Zoubin Ghahramani

clear all;
Ns = [5];
for ni=1:length(Ns)
  ndata = Ns(ni);
  
  setSeed(2);
  
  x1d=rand(ndata,1)*10; % input points
  e=randn(ndata,1); % noise
  ytrain = (x1d-3).^2 + 4*e; % actual function
  plotvals1d = [-2:0.1:10]'; % uniform grid for plotting/ testing
  trueOutput = (plotvals1d-3).^2;
  
  names = {'EB'};
  %names = {'VB'};
 
  for i=1:length(names)
    %fitFn = fitFns{i}; predFn = predFns{i};
    name = names{i};
    
    degs = [1 2 3];
    for m=1:length(degs)
      deg=degs(m);
      X = polyBasis(x1d, deg);
      X = X(:,2:end); % omit column of 1s
      Xtest = polyBasis(plotvals1d, deg);
      Xtest = Xtest(:, 2:end);
      
      [model, logev(m)] = linregFitBayes(X, ytrain, 'prior', name);
      [mu, sig2] = linregPredictBayes(model, Xtest);
      sig = sqrt(sig2);
      
      % Plot the data, the original function, and the trained network function.
      if 1
      figure;
      plot(x1d, ytrain, 'xb', 'markersize', 10, 'linewidth', 3)
      hold on
      plot(plotvals1d, trueOutput, 'g-', 'linewidth', 3);
      plot(plotvals1d, mu, 'r-.', 'linewidth', 3)
      plot(plotvals1d, mu + 2*sig, 'k:','linewidth',3);
      plot(plotvals1d, mu - 2*sig, 'k:','linewidth',3);
      title(sprintf('degree = %d', deg), 'FontSize', 24)
      xlim([-2 10]);
      set(gca, 'FontSize', 24);
      %saveas(gcf, sprintf('/home/trung/Dropbox/phdDocuments/trungthesis/figs/selectionDegree%dLarge.eps',deg),'epsc');
      saveas(gcf, sprintf('/home/trung/Dropbox/phdDocuments/trungthesis/figs/selectionDegree%dSmall.eps',deg),'epsc');
      end
    end
     
    
    figure;
    PP=exp(logev);
    PP=PP/sum(PP);
    bar(degs, PP)
    axis([-0.5 length(degs)+0.5 0 1]);
    set(gca,'FontSize',16);
    aa=xlabel('model'); set(aa,'FontSize',24);
    aa=ylabel('margnal likelihood'); set(aa,'FontSize',24);
    title(sprintf('N = %d', ndata), 'FontSize', 24)
    %saveas(gcf, ['/home/trung/Dropbox/phdDocuments/trungthesis/figs/selectionEvidenceLarge.eps'],'epsc');
    saveas(gcf, ['/home/trung/Dropbox/phdDocuments/trungthesis/figs/selectionEvidenceSmall.eps'],'epsc');
  end % for i
  
end % for ni
