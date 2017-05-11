function [MSE, NMSE, Rsquared] = my_regression_metrics( yest, y )
%MY_REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o R squared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization
M = size(yest,2);
Mu = sum(y,2)/M;
Mu_est = sum(yest,2)/M;

%Mean Square Error
MSE = sum(bsxfun(@minus,yest,y).^2,2)/M;

%Normalized Mean Square Error
NMSE = (M-1)*MSE/sum(bsxfun(@minus,y,Mu).^2,2);

%Coefficient of Determination R
num = bsxfun(@minus,y,Mu)*bsxfun(@minus,yest,Mu_est)';
denum1 = (sum(bsxfun(@minus,y,Mu).^2,2))^0.5;
denum2 = (sum(bsxfun(@minus,yest,Mu_est).^2,2))^0.5;

Rsquared = (num/(denum1*denum2))^2;

end

