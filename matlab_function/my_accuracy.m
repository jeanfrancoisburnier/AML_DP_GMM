function [acc] =  my_accuracy(y_test, y_est)
%My_accuracy Computes the accuracy of a given classification estimate.
%   input -----------------------------------------------------------------
%   
%       o y_test  : (1 x M_test),  true labels from testing set
%       o y_est   : (1 x M_test),  estimated labes from testing set
%
%   output ----------------------------------------------------------------
%
%       o acc     : classifier accuracy
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialisation
M_test = size(y_test,2);
d=zeros(1,M_test);

%Check the number points of y_test are equals to y_est
for i = 1 : M_test
    if(y_test(i)==y_est(i))
        d(i) = 1 ;
    end
end

%evaluation of accuracy
acc = sum(d)/M_test;

end