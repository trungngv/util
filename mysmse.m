function smse = mysmse(ytrue, ypred, trainmean)
%MYSMSE smse = mysmse(ytrue, ypred)
%   Compute the standardised mean square error (SMSE), also NMSE in some
%   publications.
% 
% SMSE = mean square error / mean test variance
smse=mean((ypred-ytrue).^2)/mean((trainmean-ytrue).^2);
end

