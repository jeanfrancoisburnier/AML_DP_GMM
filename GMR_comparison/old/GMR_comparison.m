clc
clearvars
close all
data_source = 0;
n_loop = 5;
n_iter = 5;
time = zeros(n_loop,1);
error = zeros(n_loop,1);
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
%                                 DP_GMM                                  % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lP_max = -1000000;
for l = 1:n_loop
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                 DP_GMM                                  % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    for k = 1:n_iter
        % run the CRP sampler to generate the posterior distribution over model 
        % parameters
        [class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(X_train, 200,1);
        
        [max_p, pos] = max(lP_record);
        if (max_p > lP_max)
            id_best = class_id(:,pos);
            mean_best = mean_record(pos);
            covar_best = covariance_record(pos);
            k_best = K_record(pos)-1;
        end
    end
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                             DP_GMR Gaussians                            % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Priors = zeros(1, k_best);
    Mu = zeros(data_size(1), k_best);
    Sigma = zeros(data_size(1),data_size(1), k_best);
    Mu_DP = cell2mat(mean_best);
    Sigma_DP = cell2mat(covar_best);
    for i = 1:k_best
        class = id_best;
        X_k = X_train(:,class == i);
        X_k_c = zeros(data_size(1),size(X_k,2));
        Priors(i) = size(X_k,2)/size(X_train,2);
        Mu(:,i) = mean(X_k,2);
        if (size(X_k,2) > data_size(1));
            for j = 1:data_size(1)
                X_k_c(j,:) = X_k(j,:) - Mu(j,i);
            end
            Sigma(:,:,i) = X_k_c*X_k_c'/(size(X_k,2)-1);
        else
            Sigma(:,:,i) = Sigma_DP(:,:,i);
        end
    end

    time(l) = toc;
    if (data_size(1) == 2)
        %ml_plot_gmm_pdf(X, Priors, Mu, Sigma)
    end

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                           Regression                                    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    in = 1: data_size(1)-1; 
    out = data_size(1);

    %x = linspace(min(X(1,:)),max(X(1,:)),300);
    x = X(1:end-1,:);
    [y_est, Sigma_y] = ml_gmr(Priors, Mu, Sigma, x, in, out);

    [y_est_test, Sigma_y_test] = ml_gmr(Priors, Mu, Sigma, X_test(1:end-1,:), in, out);

    if data_source == 0
        error(l) = var(y_est_test'-y_test);
    else
        error(l) = var(y_est_test-X_test(end,:));
    end
    if (size(X,1)<3)
        ml_plot_gmr_function(x', y_est, Sigma_y,'var_scale');
    end
end
avg_time = mean(time)
avg_error = mean(error)