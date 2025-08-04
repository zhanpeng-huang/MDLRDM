%% 加载数据
clear
clc
warning off all
addpath(genpath('./'))
addpath("Functions\")
%%加载数据，归一化
load 3sources.mat
x1=data{1};
x2=data{2};
x3=data{3};
truth = truelabel{1}';
for l = 1 : size(x1,2)
    x1(:,l) = x1(:,l)/norm(x1(:,l));
end
for l = 1 : size(x2,2)
    x2(:,l) = x2(:,l)/norm(x2(:,l));
end
for l = 1 : size(x3,2)
    x3(:,l) = x3(:,l)/norm(x3(:,l));
end
X{1} = x1;
X{2} = x2;
X{3} = x3;
%%存储结果
i1=1;
j1=1;
k1=1;
L1 = 1;
ACC_mean= [];
NMI_mean= [];
AR_mean= [];
Fscore_mean= [];
Purity_mean= [];
Precision_mean= [];
Recall_mean= [];
Entropy_mean= [];

ACC_std= [];
NMI_std= [];
AR_std= [];
Fscore_std= [];
Purity_std= [];
Precision_std= [];
Recall_std= [];
Entropy_std= [];

timer_all = [];
%%设置参数
K1 = length(unique(truth));
k=floor(size(x1,2)/10);
Alpha  = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
Alpha  = [0.6];

Lambda1 = [0.001 0.01 0.1 1 10 100 1000];
Lambda1 = [1];

Lambda2 = [0.001 0.01 0.1 1 10 100 1000];
Lambda2 = [0.001];

for i = 1:length(Alpha)
    alpha = Alpha(i)
    for ii = 1:length(Lambda1)
        lambda1 = Lambda1(ii)
        for iii = 1:length(Lambda2)
            lambda2 = Lambda2(iii)
            tic
            W = MDLRDM(X,lambda1,lambda2,k,alpha);
            group = SpectralClustering2(W,K1);
            Results = Clustering8Measure(truth, group);
            timer = toc;
            for km = 1:10
                Results(km, :) = Clustering8Measure(truth, group); % 存储所有指标
                result_mean = mean(Results(1:km, :), 1);
                results_std = std(Results(1:km, :), 1);
            end
            %           result = [ACC nmi AR Fscore Purity Precision Recall Entropy];
            result_mean
            if i1<=length(Lambda1)
                ACC_mean(i1,j1) = result_mean(1);
                NMI_mean(i1,j1) = result_mean(2);
                AR_mean(i1,j1) = result_mean(3);
                Fscore_mean(i1,j1) = result_mean(4);
                Purity_mean(i1,j1) = result_mean(5);
                Precision_mean(i1,j1) = result_mean(6);
                Recall_mean(i1,j1) = result_mean(7);
                Entropy_mean(i1,j1) = result_mean(8);


                ACC_std(i1,j1) = results_std(1);
                NMI_std(i1,j1) = results_std(2);
                AR_std(i1,j1) = results_std(3);
                Fscore_std(i1,j1) = results_std(4);
                Purity_std(i1,j1) = results_std(5);
                Precision_std(i1,j1) = results_std(6);
                Recall_std(i1,j1) = results_std(7);
                Entropy_std(i1,j1) = results_std(8);
                timer_all(i1,j1) = timer;
                j1=j1+1;
            else
                i1=k1;
                ACC_mean(i1,j1) = result_mean(1);
                NMI_mean(i1,j1) = result_mean(2);
                AR_mean(i1,j1) = result_mean(3);
                Fscore_mean(i1,j1) = result_mean(4);
                Purity_mean(i1,j1) = result_mean(5);
                Precision_mean(i1,j1) = result_mean(6);
                Recall_mean(i1,j1) = result_mean(7);
                Entropy_mean(i1,j1) = result_mean(8);

                ACC_std(i1,j1) = results_std(1);
                NMI_std(i1,j1) = results_std(2);
                AR_std(i1,j1) = results_std(3);
                Fscore_std(i1,j1) = results_std(4);
                Purity_std(i1,j1) = results_std(5);
                Precision_std(i1,j1) = results_std(6);
                Recall_std(i1,j1) = results_std(7);
                Entropy_std(i1,j1) = results_std(8);
                timer_all(i1,j1) = timer;

                j1=j1+1;
            end
        end
        j1=1;
        i1=i1+1;
        k1=k1+1;
    end
    k1=i1;
end
% save('./result/3sources_result','ACC_mean','NMI_mean','AR_mean','Fscore_mean', ...
%     'Purity_mean','Precision_mean','Recall_mean',"Entropy_mean",'ACC_std','NMI_std','AR_std','Fscore_std', ...
%     'Purity_std','Precision_std','Recall_std',"Entropy_std","timer_all")
