function [y_est] = my_gmm_classif(X_test, models, labels, K, P_class)
%MY_GMM_CLASSIF Classifies datapoints of X_test using ML Discriminant Rule
%   input------------------------------------------------------------------
%
%       o X_test    : (N x M_test), a data set with M_test samples each being of
%                           dimension N, each column corresponds to a datapoint.
%       o models    : (1 x N_classes) struct array with fields:
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o labels    : (1 x N_classes) unique labels of X_test.
%       o K         : (1 x 1) number K of GMM components.
%   optional---------------------------------------------------------------
%       o P_class   : (1 x N_classes), the vector of prior probabilities
%                      for each class i, p(y=i). If provided, equal class
%                      distribution assumption is no longer made.
%
%   output ----------------------------------------------------------------
%       o y_est  :  (1 x M_test), a vector with estimated labels y \in {0,...,N_classes}
%                   corresponding to X_test.
%%
%initialization
M_test = size(X_test,2);
N_classes = size(models,2);
ll=zeros(M_test,N_classes);
y_est=zeros(1,M_test);

%default parameters : if equal distribution of classes assumption
if nargin<5
    P_class = ones(1,N_classes);
end

for i = 1 : M_test
    for j = 1 : N_classes
        %compute loglikelyhood
        ll(i,j) = -my_gmmLogLik(X_test(:,i), models(j).Priors, models(j).Mu, models(j).Sigma)-log(P_class(j));
    end
    % ML discriminant rule
    [~, c] = min(ll(i,:));
    y_est(i)=c-1;
end

end