clc
clearvars
close all
data_source = 0;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               LOAD DATASET                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if data_source == 1
    addpath('matlab_function');
    addpath('Datasets');
    addpath(genpath('ML_toolbox-master'));

    %load('2D-GMM.mat')
    load('2D-GMM.mat')

    % Visualize Dataset
    options.class_names = {};
    options.title       = '2D Concentric Circles Dataset';
    %options.labels       = y;

    if exist('h0','var') && isvalid(h0), delete(h0);end
    h0 = ml_plot_data(X',options);hold on;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               LOAD DATASET                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if data_source == 0
    [X,y,y_noisy] = load_regression_datasets('1d-sine');

    X = [X, y_noisy]';
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          define training set                            % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 0.5; %define training/test ratio
data_size = size(X);
%select = randsrc(1,data_size(2),[0 1; (1-p) p]);

%X_temp = zeros(data_size(1),data_size(2));
%X_train = zeros(data_size(1),sum(select));
%X_test = zeros(data_size(1),data_size(2)-sum(select));
%for i = 1:data_size(1)
%    X_temp(i,:) = X(i,:).* select;
%    temp = X_temp(i,:);
%    temp(temp == 0) = [];
%    X_train(i,:) = temp;
%    X_temp(i,:) = X(i,:).* (~select);
%    temp = X_temp(i,:);
%    temp(temp == 0) = [];
%    X_test(i,:) = temp;
%end

   % determine how many elements is ten percent
   numelements = round(p*data_size(2));
   % get the randomly-selected indices
   indices = randperm(data_size(2));
   % choose the subset of a you want
   X_train = X(:,indices(1:numelements));
   X_test = X(:,indices(numelements+1:end));

% show the training data
figure(2)
plot_mixture(X_train, ones(1,size(X_train,2)))

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 DP_GMM                                  % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% run the CRP sampler to generate the posterior distribution over model 
% parameters
tic;
[class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(X_train, 200,8);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             DP_GMR Gaussians                            % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Priors = zeros(1, K_record(end)-1);
Mu = zeros(data_size(1), K_record(end)-1);
Sigma = zeros(data_size(1),data_size(1), K_record(end)-1);
Mu_DP = cell2mat(mean_record(end));
Sigma_DP = cell2mat(covariance_record(end));
for i = 1:K_record(end)-1
    class = class_id(:,end);
    X_k = X_train(:,class == i);
    X_k_c = zeros(data_size(1),size(X_k,2));
    Priors(i) = size(X_k,2)/size(X_train,2);
    Mu(:,i) = mean(X_k,2);
    if (size(X_k,2) ~= 1);
        for j = 1:data_size(1)
            X_k_c(j,:) = X_k(j,:) - Mu(j,i);
        end
        Sigma(:,:,i) = X_k_c*X_k_c'/(size(X_k,2)-1);
    else
        Sigma(:,:,i) = Sigma_DP(:,:,i);
    end
end

time = toc
if (data_size(1) == 2)
    ml_plot_gmm_pdf(X, Priors, Mu, Sigma)
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               EM Gaussians                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% From the previous graph choose the best value of K
%K = K_record(end)-1; cov_type = 'full';  plot_iter = 0;

% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
%tic;
%[Priors, Mu, Sigma] = ml_gmmEM(X, K);
%toc;

% Visualize GMM pdf from learnt parameters
%close all;
%if (data_size(1) == 2)
%    ml_plot_gmm_pdf(X, Priors, Mu, Sigma)
%end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Regression                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in = 1: data_size(1)-1; 
out = data_size(1);

%x = linspace(min(X(1,:)),max(X(1,:)),300);
x = X(1:end-1,:);
[y_est, Sigma_y] = ml_gmr(Priors, Mu, Sigma, x, in, out);
if data_source == 0
    error = var(y_est-y)
else
    error = var(y_est-y)
end
ml_plot_gmr_function(x', y_est, Sigma_y,'var_scale');