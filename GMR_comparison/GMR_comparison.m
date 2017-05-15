clc
clearvars
close all
data_source = 0;
n_loop = 40;
n_iter = 5;
time = zeros(n_loop,1);
MSE = zeros(n_loop,1);NMSE = zeros(n_loop,1);Rsquared = zeros(n_loop,1);
a_0 = logspace(-2,2,5);
b_0 = logspace(-2,2,5);
%a_0 = 100;
%b_0 = 0.01;
v = 1;
w = 1;
[a_param, b_param] = meshgrid(a_0,b_0);
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
    [X,y,y_noisy] = load_regression_datasets('1d-sine');

    X = [X, y_noisy]';
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               PARAM LOOP                                % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lp1 = length(a_0);
lp2 = length(b_0);
P = [0.25, 0.5, 0.75];
avg_time = zeros(lp1,lp2);
avg_mse = zeros(lp1,lp2);
avg_nmse = zeros(lp1,lp2);
avg_rsquared = zeros(lp1,lp2);
var_time = zeros(lp1,lp2);
var_mse = zeros(lp1,lp2);
var_nmse = zeros(lp1,lp2);
var_rsquared = zeros(lp1,lp2);
quantile_time = zeros(lp1,lp2,length(P));
quantile_mse = zeros(lp1,lp2,length(P));
quantile_nmse = zeros(lp1,lp2,length(P));
quantile_rsquared = zeros(lp1,lp2,length(P));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             gridsearch                                  % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for v = 1:lp1
    for w = 1:lp2
        v
        w
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
        %figure(2)
        %plot_mixture(X_train, ones(1,size(X_train,2)))

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
                [class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(X_train, 500,a_0(v),b_0(w));

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
            %Mu = cell2mat(mean_best);
            %Sigma = cell2mat(covar_best);
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
                [MSE(l), NMSE(l), Rsquared(l)] = my_regression_metrics( y_est_test, y_test');
            else
                [MSE(l), NMSE(l), Rsquared(l)] = my_regression_metrics( y_est_test', X_test(end,:) );
            end
            if (size(X,1)<3)
                %ml_plot_gmr_function(x', y_est, Sigma_y,'var_scale');
            end
        end
        avg_time(v,w) = mean(time)/n_iter;
        avg_mse(v,w) = mean(MSE);
        avg_nmse(v,w) = mean(NMSE);
        avg_rsquared(v,w) = mean(Rsquared);
        var_time(v,w) = var(time/n_iter);
        var_mse(v,w) = var(MSE);
        var_nmse(v,w) = var(NMSE);
        var_rsquared(v,w) = var(Rsquared);
        quantile_time(v,w,:) = quantile(time/n_iter,P);
        quantile_mse(v,w,:) = quantile(MSE,P);
        quantile_nmse(v,w,:) = quantile(NMSE,P);
        quantile_rsquared(v,w,:) = quantile(Rsquared,P);
    end
end
%% Plot
figure('color', [1 1 1]);
surf(a_param, b_param, quantile_time(:,:,2))
xlabel('a_0')
ylabel('b_0')
zlabel('time [s]')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, avg_time)
xlabel('a_0')
ylabel('b_0')
zlabel('time [s]')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, quantile_mse(:,:,2))
xlabel('a_0')
ylabel('b_0')
zlabel('mse')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, avg_mse)
xlabel('a_0')
ylabel('b_0')
zlabel('mse')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, quantile_nmse(:,:,2))
xlabel('a_0')
ylabel('b_0')
zlabel('nmse')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, avg_nmse)
xlabel('a_0')
ylabel('b_0')
zlabel('nmse')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, quantile_rsquared(:,:,2))
xlabel('a_0')
ylabel('b_0')
zlabel('R squared')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
figure('color', [1 1 1]);
surf(a_param, b_param, avg_rsquared)
xlabel('a_0')
ylabel('b_0')
zlabel('R squared')
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');