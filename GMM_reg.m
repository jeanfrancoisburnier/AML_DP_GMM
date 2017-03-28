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
% 
%Set GMM-hyperparameters
K = 1; cov_type = 'full'; plot_iter=1;
[  Priors, Mu, Sigma ] = ml_gmmEM(X, K);
Sigma  = ml_covariance( X, Mu, type )
prob = ml_gaussPDF(Data, Mu, Sigma);

%[ loglik ] = ml_LogLikelihood_gmm(X,Priors,Mu,Sigma)
%[y, Sigma_y, beta] = ml_gmr(Priors, Mu, Sigma, x, in, out)