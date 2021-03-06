clc
clearvars
close all
data_source = 1;

n_loop = 1;
time = zeros(n_loop,1);
%error = zeros(n_loop,1);
MSE = zeros(n_loop,1);NMSE = zeros(n_loop,1);Rsquared = zeros(n_loop,1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               LOAD DATASET                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if data_source == 1
    addpath('matlab_function');
    addpath('Datasets');
    addpath(genpath('ML_toolbox-master'));

    %load('2D-GMM.mat')
    load('RedWine_Quality.mat')
    if (size(X,1)<4)
        % Visualize Dataset
        options.class_names = {};
        options.title       = '2D Concentric Circles Dataset';
        %options.labels       = y;

        if exist('h0','var') && isvalid(h0), delete(h0);end
        h0 = ml_plot_data(X',options);hold on;
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               LOAD DATASET                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if data_source == 0
    [X,y,y_noisy] = load_regression_datasets('1d-sinc');

    X = [X, y_noisy]';
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               PARAM LOOP                                % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = [0.25, 0.5, 0.75];
repeat = 20;
avg_time = zeros(repeat);
avg_mse = zeros(repeat);
avg_nmse = zeros(repeat);
avg_rsquared = zeros(repeat);
var_time = zeros(repeat);
var_mse = zeros(repeat);
var_nmse = zeros(repeat);
var_rsquared = zeros(repeat);
quantile_time = zeros(repeat,length(P));
quantile_mse = zeros(repeat,length(P));
quantile_nmse = zeros(repeat,length(P));
quantile_rsquared = zeros(repeat,length(P));


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          define training set                            % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 0.75; %define training/test ratio
data_size = size(X);
   % determine how many elements is ten percent
   numelements = round(p*data_size(2));
   % get the randomly-selected indices
   indices = randperm(data_size(2));
   % choose the subset of a you want
   X_train = X(:,indices(1:numelements));
   X_test = X(:,indices(numelements+1:end));
   if data_source == 0
       y_test = y(indices(numelements+1:end));
   end

% show the training data
figure(2)
plot_mixture(X_train, ones(1,size(X_train,2)))



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Testing dfferent K on gmm_eval.m                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % K-means Evaluation Parameters
% K_range=[1:10]; cov_type = 'full'; repeats = 10;
% 
% % Evaluation of gmm-em in order to find the optimal k
% for l = 1:n_loop
%     tic;
%     gmm_eval(X, K_range, repeats, cov_type);
%     time(l) = toc;
% end
% mean(time)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Choice of the GMM-hyperparameters                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for l = 1:repeat
    % From the previous graph choose the best value of K
    K = 5; cov_type = 'full';  plot_iter = 0;

    % Run MY GMM-EM function, estimates the paramaters by maximizing loglik
    [Priors, Mu, Sigma] = ml_gmmEM(X, K);

    % Visualize GMM pdf from learnt parameters
    close all;
    %ml_plot_gmm_pdf(X, Priors, Mu, Sigma)

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           Regression                                    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    in = 1: data_size(1)-1; 
    out = data_size(1);

    %x = linspace(-30,40,300);
    x = X(1:end-1,:);
    [y_est, Sigma_y] = ml_gmr(Priors, Mu, Sigma, x, in, out);
    [y_est_test, Sigma_y_test] = ml_gmr(Priors, Mu, Sigma, X_test(1:end-1,:), in, out);

    if data_source == 0
        [MSE(l), NMSE(l), Rsquared(l)] = my_regression_metrics( y_est_test, y_test');
    else
        [MSE(l), NMSE(l), Rsquared(l)] = my_regression_metrics( y_est_test, X_test(end,:) );
    end

    ml_plot_gmr_function(x', y_est, Sigma_y,'var_scale');
    
end
%%
avg_mse = mean(MSE);
avg_nmse = mean(NMSE);
avg_rsquared = mean(Rsquared);
%var_time = var(time/n_iter);
var_mse = var(MSE);
var_nmse = var(NMSE);
var_rsquared = var(Rsquared);
%quantile_time = quantile(time/n_iter,P);
quantile_mse = quantile(MSE,P);
quantile_nmse = quantile(NMSE,P); 

quantile_rsquared = quantile(Rsquared,P);