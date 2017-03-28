clc
clear all
close all

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               LOAD DATASET                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('GMM-Datasets/2d-concentric-circles.mat')

% Visualize Dataset
options.class_names = {};
options.title       = '2D Concentric Circles Dataset';
options.labels       = y;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Testing dfferent K on gmm_eval.m                           %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means Evaluation Parameters
K_range=[1:10]; cov_type = 'iso'; repeats = 10;

% Evaluation of gmm-em in order to find the optimal k
gmm_eval(X, K_range, repeats, cov_type);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% Pick best K from Plot and Visualize result

% Set GMM Hyper-parameters <== CHANGE VALUES HERE!
K = 3; cov_type = 'full';  plot_iter = 0;

%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
tic;
[Priors, Mu, Sigma] = ml_gmmEM(X, K, cov_type,  plot_iter);
toc;

% Compute GMM Likelihood
[ ll ] = my_gmmLogLik(X, Priors, Mu, Sigma);

% Visualize GMM pdf from learnt parameters
close all;
ml_plot_gmm_pdf(X, Priors, Mu, Sigma)

%Set GMM-hyperparameters
K = 1; cov_type = 'full'; plot_iter=1;
[  Priors, Mu, Sigma ] = ml_gmmEM(X, K);
Sigma  = ml_covariance( X, Mu, cov_type )
prob = ml_gaussPDF(X, Mu, Sigma);

[ loglik ] = ml_LogLikelihood_gmm(X,Priors,Mu,Sigma)
%[y, Sigma_y, beta] = ml_gmr(Priors, Mu, Sigma, X, in, out)