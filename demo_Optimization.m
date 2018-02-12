load('A.mat'); % It can be any undirected positive-weighted network with 0<=w_ij<=1


% figure,plot(graph(A));
figure,imagesc(A),colorbar;
%%
% options for the four conflicr measures:
m = 2; % [1 for the internal conflict - ic; 
%         2 for the external conflict - ec; 
%         3 for the controversy - c; 
%         4 for the resistance - r]

% options for the two methods
gradient = 1; % [1 for projected gradient descent; 
%                0 for coordinate descent]

% options for both case internal opinions
avgCase = 1; % [1 for average case s; 
%               0 for the worst case s]

iter = 50;
k = 6; % k for k/2 edges
stepsz = 1;
dim = 10;

[OptA, acr, wcr, conflicts] = ConflictRiskOptimization(A,m,gradient,avgCase,iter,k,stepsz,dim);



