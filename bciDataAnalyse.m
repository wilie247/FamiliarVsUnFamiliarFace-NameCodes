clc; 
clear all; close all;
 
 load("cVEP_dataset/AA002.mat")

 


 
 
 all_labels = data_y';
 feat = data_x;
 %get filtered signal alpha signal
fs = 128;
a = delta(feat(:,1:14), fs); 
b = theta(feat(:,1:14), fs); 
c = alpha(feat(:,1:14), fs); 
d = beta(feat(:,1:14), fs); 


%d(1) = a; d(2) = 

% ts=128;
% t=1:1000/ts:size(data, 1)*(1000/ts);
% ch = 1;
% % 
% subplot(4, 2, 1), plot(t, a), title("Delta " + ch);
% subplot(4, 2, 3), plot(t, b), title("Theta " + ch);
% subplot(4, 2, 4), plot(t, c), title("Alpha " + ch);
% subplot(4, 2, 5), plot(t, d), title("Beta " + ch);

 %call Pnn

for i=1:4
    %get
    fs = 128;
    if i == 1
        feat = delta(feat(:,1:14), fs); 
    elseif i==2
        feat = theta(feat(:,1:14), fs); 
    elseif i==2
        feat = alpha(feat(:,1:14), fs);
    elseif i==2
        feat = beta(feat(:,1:14), fs); 
    end
    % (1) Perform k-nearest neighbor (KNN)
    disp("The class with data " + i)
    
    
    
    %call Pnn
    kfold = 10;
    FPnn = Fpnn(feat, label,kfold, i);
    
    k=3; % k-value in KNN
    KNN=jKNN(feat,label,k,kfold) 


    % (2) Perform discriminate analysis (DA)
    kfold=10; Disc='l'; % The Discriminate can selected as follows:
    % 'l' : linear 
    % 'q' : quadratic
    % 'pq': pseudoquadratic
    % 'pl': pseudolinear
    % 'dl': diaglinear
    % 'dq': diagquadratic
    DA=jDA(feat,label,Disc,kfold) 

    % (3) Perform Naive Bayes (NB)
    kfold=10; Dist='n'; % The Distribution can selected as follows:
    % 'n' : normal distribution 
    % 'k' : kernel distribution
    NB=jNB(feat,label,Dist,kfold) 

    % (4) Perform decision tree (DT)
     kfold=10; nSplit=50; % Number of split in DT
     DT=jDT(feat,label,nSplit,kfold)

    % (5) Perform support vector machine (SVM with one versus one)
    kfold=10; kernel='r'; % The Kernel can selected as follows:
    % 'r' : radial basis function  
    % 'l' : linear function 
    % 'p' : polynomial function 
    % 'g' : gaussian function
    %SVM=jSVM(feat,label,kernel,kfold)
    % (6) Perform random forest (RF)
    %kfold=10; nBag=50; % Number of bags in RF
    %RF=jRF(feat,label,nBag,kfold)


    % (6) Perform ETMEST Classification
    % col = column of data holding class labels
    % 
    % hold data for testing supply decimal e.g 0.4 for 40 percent
    % col = 15; hold = 0.4;
    % myETMEST=etmest(data,col,hold)


    %use the predict method to get the performance on new data
    % Use essemble method and let majority carry the winner.

    % [label,score] = predict(ClassificationECOC, XTest);

    %spectogram image and classification
    
    

end