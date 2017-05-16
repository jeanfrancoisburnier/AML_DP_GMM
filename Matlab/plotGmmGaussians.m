function plotGmmGaussians(X,Mu,Sigma,sig)

    n = size(Mu,2);
    
    % Visualize Dataset
    options.class_names = {};
    options.title       = 'Gaussian Mixture Model';
    options.colors      = [0,1,0];
    %options.labels       = y;

    if exist('h0','var') && isvalid(h0), delete(h0);end
    h0 = ml_plot_data(X',options);hold on;
    
    for k = 1:n
        plot_gaussian_ellipsoid(Mu(:,k),Sigma(:,:,k),sig);
        %plot(Mu(1,k),Mu(2,k),'o');
    end
    
    color_centroid = repmat([0,0,1],n,1);
    ml_plot_centroid(Mu',color_centroid);
    
    xlabel({'$x$'}, 'Interpreter','Latex','FontSize',18,'FontName','Times', 'FontWeight','Light');
    ylabel({'$y$'}, 'Interpreter','Latex','FontSize',18,'FontName','Times', 'FontWeight','Light');
    
    hold off;
    
end