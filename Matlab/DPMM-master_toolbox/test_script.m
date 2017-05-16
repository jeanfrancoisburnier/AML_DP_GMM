clc; clear all; close all;

mu(:,1) = [-10,-10];
sigSq(:,:,1) = 0.01*eye(2);

%mu(:,2) = [10,10]';
%sigSq(:,:,2) = eye(2);

[Y,z,mu,ss,p] = drawGmm(10,mu,sigSq);
%subplot(1,2,1);
title('generative clusters');
scatterMixture(Y,z);
%params = dpmm(Y,100);
%subplot(1,2,2);
%title('dpmm clustering');
%scatterMixture(Y,params(end).classes);
