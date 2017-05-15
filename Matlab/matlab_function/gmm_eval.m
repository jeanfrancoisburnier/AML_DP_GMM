function [] =  gmm_eval(X, K_range, repeats, cov_type,init_type)
%GMM_EVAL Implementation of the GMM Model Fitting with AIC/BIC metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o cov_type : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization
AIC_curve = zeros(1, length(K_range));
BIC_curve = zeros(1, length(K_range));

plot_iter = 0;

% Populate Curves
for i=1:length(K_range)
    
    % Select K from K_range
    K = K_range(i); 
    
    % Repeat k-means X times
    AIC_ = zeros(1, repeats); BIC_= zeros(1, repeats);     
    for ii = 1:repeats
        [ Priors, Mu, Sigma ] = ml_gmmEM(X, K,init_type, cov_type);
        AIC_(ii) = ml_gmm_aic(X, Priors,Mu,Sigma,cov_type);
        BIC_(ii) = ml_gmm_bic(X,Priors,Mu,Sigma,cov_type);
    end 
    
    % Get the mean of those X repeats
    AIC_curve(i) = mean(AIC_);
    BIC_curve(i) = mean(BIC_);
    
end

% Plot Metric Curves
figure;
plot(AIC_curve,'--o', 'LineWidth', 1); hold on;
plot(BIC_curve,'--o', 'LineWidth', 1); hold on;
xlabel('K')
legend('AIC', 'BIC')
title(sprintf('GMM (%s) Model Fitting Evaluation metrics',cov_type))
grid on

end