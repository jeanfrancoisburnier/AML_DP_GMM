function [handle] = ml_plot_data(X,options)
%ML_PLOT_DATA Simple wrapper to plot X data
%
%   input -----------------------------------------------------------------
%
%       o X : (N x [1,2,3]), Dataset of N samples which can be one, two or
%                            three dimensional.
%
%       o options: struct,  
%           options.labels      : (N x 1), class labels
%           options.is_eigen    : if it is eigenvector space or not.
%           options.weights     : (N x 1), weights of data points (optional)
%
%   output ----------------------------------------------------------------
%
%       o handle : handle to the figure
%
%
%
%% Input processing




[~,D] = size(X);



%% Extract Options [labels,is_eig,etc..]

labels              = [];
class_names         = [];
colors              = [];
title_name          = 'My pretty data';
is_eig              = false;
label_font_size     = 14;
points_size         = 50;
weights             = [];
plot_labels         = [];
plot_figure         = false; % if true a new figure will not be created.

if exist('options','var')
   if isfield(options,'labels'),        labels      = options.labels;       end
   if isfield(options,'is_eig'),        is_eig      = options.is_eig;       end
   if isfield(options,'title'),         title_name  = options.title;        end
   if isfield(options,'points_size'),   points_size = options.points_size;  end
   if isfield(options,'class_names'),   class_names = options.class_names;  end
   if isfield(options,'weights'),       weights     = options.weights;      end
   if isfield(options,'colors'),         colors     = options.colors;       end
   if isfield(options,'plot_figure'),   plot_figure   = options.plot_figure;  end
   if isfield(options,'plot_labels'),   plot_labels   = options.plot_labels;  end
end

if ~isempty(labels) && isempty(colors)
   colors = hsv(length(unique(labels)));
end

if ~isempty(labels)
    points_size = repmat(points_size,length(labels),1);
end

if ~isempty(weights)
    points_size = weights;
end


%% Plot the data

if plot_figure == true
    handle = [];
else
    handle = figure;
end

% if sum(labels < 1) ~= 0 % some labels are negative
%     labels_tmp   = zeros(size(labels));
%     labels_class = unique(labels);
%     
%     for i=1:length(labels_class)
%         labels_tmp(labels_class(i)==labels) = i;
%     end
%     labels = labels_tmp;
% end

set(gca,'FontSize',14);
hold on;

if (is_eig)
    
    if ~isempty(labels)
        gplotmatrix(X,[],labels,colors,'.',10);
        h = findobj('Tag','legend');
        set(h, 'String',class_names);
    else
        gplotmatrix(X,[],ones(size(X,1),1));
    end
    box on; grid on;
    
else    
    
    if D == 1
        plot(X,'o');
        xlabel('x');
        if ~isempty(colors)
            scatter(X,zeros(1,length(X)),points_size,colors(labels,:));
        else
            scatter(X,zeros(1,length(X)),points_size);
        end
    elseif D == 2
        if ~isempty(labels)
            id_labels = unique(labels);
            for i=1:size(colors,1)
                idx   = labels == id_labels(i);
                scatter(X(idx,1),X(idx,2),points_size(idx),'filled','MarkerFaceColor',colors(i,:),'MarkerEdgeColor', [0 0 0]);
            end
        elseif ~isempty(colors)
            scatter(X(:,1),X(:,2),points_size,colors,'filled','MarkerEdgeColor', [0 0 0]);
        else
            scatter(X(:,1),X(:,2),points_size,'filled','MarkerEdgeColor', [0 0 0]);
        end
        
    elseif D == 3
        
        if ~isempty(colors)
            id_labels = unique(labels);
            for i=1:size(colors,1)
                idx   = labels == id_labels(i);
                scatter3(X(idx,1),X(idx,2),X(idx,3),points_size(idx),'filled','MarkerFaceColor',colors(i,:),'MarkerEdgeColor', [0 0 0]);
            end
        else
            scatter3(X(:,1),X(:,2),X(:,3),points_size,'filled');
        end
        
    else
        if ~isempty(labels)
            gplotmatrix(X,[],labels,colors,'.',12);
            h = findobj('Tag','legend');
            set(h, 'String',class_names);
        else
            gplotmatrix(X,[],ones(size(X,1),1));
        end        
    end    
end   

if ~isempty(class_names) && D < 4 && is_eig == false
   legend(class_names{:},'FontName','Times'); 
end

hold off;
box on; grid on;
%% Set title 

if plot_figure==false, 
    title(title_name, 'Interpreter','tex','FontName','Times', 'FontWeight','Light'); 
end

%% Set the labels
if is_eig == true && D <= 3
        if D < 2
            xlabel('eig1','FontSize',label_font_size);
        end
        if D < 3
            xlabel('eig2','FontSize',label_font_size);
            ylabel('eig1','FontSize',label_font_size);
        end
        if D == 3
%             zlabel('eig3','FontSize',label_font_size);
        end
elseif D <= 3
    
        if isempty(plot_labels)
            if D >= 1
                xlabel('x','FontSize',label_font_size);
            end
            if D >= 2
                ylabel('y','FontSize',label_font_size);
            end
            if D == 3
                zlabel('z','FontSize',label_font_size);
            end
        else
            xlabel(plot_labels{1},'FontSize',label_font_size,'FontName','Times', 'FontWeight','Light');
            ylabel(plot_labels{2},'FontSize',label_font_size,'FontName','Times', 'FontWeight','Light');
            zlabel(plot_labels{3},'FontSize',label_font_size,'FontName','Times', 'FontWeight','Light');
        end        
end


if D ==3
    view(3)
end
end



