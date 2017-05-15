clc
clearvars
close all

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               LOAD DATASET                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('matlab_function');
addpath('Datasets');
addpath(genpath('ML_toolbox-master'));

load('2D-GMM.mat')
[X,y,y_noisy]=load_regression_datasets('1d-sine');
X = [X , y_noisy]';
%load('s.mat')

% Visualize Dataset
options.class_names = {};
options.title       = '2D Concentric Circles Dataset';
%options.labels       = y;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Testing dfferent K on gmm_eval.m                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means Evaluation Parameters
K_range=[1:10]; cov_type = 'iso'; repeats = 10; init_type = 'uniform'

% Evaluation of gmm-em in order to find the optimal k
gmm_eval(X, K_range, repeats, cov_type,init_type);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Choice of the GMM-hyperparameters                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% From the previous graph choose the best value of K
K = 3; cov_type = 'iso';  plot_iter = 0;

% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
tic;
[Priors, Mu, Sigma] = ml_gmmEM(X, K, init_type);
toc;

% Visualize GMM pdf from learnt parameters
close all;
ml_plot_gmm_pdf(X, Priors, Mu, Sigma)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Regression                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in = 1; out = 2;

x = linspace(-30,40,300);
[y_est, Sigma_y] = ml_gmr(Priors, Mu, Sigma, x, in, out);

ml_plot_gmr_function(x', y_est, Sigma_y,'var_scale');
