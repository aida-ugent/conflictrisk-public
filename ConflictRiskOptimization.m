% ConflictRiskOptimization.m 
% Description: 
% Given the adjacency matrix of a social network, this code optimizes 
% the average-case conflict risk (ACR) and 
% the worst-case conflict risk (WCR). 

%% load data
% adjacency matrix of the network as input.

% load('A.mat'); % It can be any undirected positive-weighted network with 0<=w_ij<=1

% n = 50;m = 95;
A = random_graph(50,0.05); % erdos renyi random network of node n, and edge m
% A = full(scale_free(n, 10, 1)); % barabasi albert model ()
% G = WattsStrogatz(n,1,1); A = full(adjacency(G)); % Watts-Strogatz model

% observe the edge edits
A = full(A);
figure,plot(graph(A));
figure,imagesc(A),colorbar;

%% parameters
% options for both case internal opinions
avgCase = 0; % [1 for average case s; 
%               0 for the worst case s]

% options for the two methods
gradient = 0; % [1 for projected gradient descent; 
%                0 for coordinate descent]

% options for the four conflicr measures:
m = 2; % [1 for the internal conflict - ic; 
%         2 for the external conflict - ec; 
%         3 for the controversy - c; 
%         4 for the resistance - r]

%% original network 
L = diag(sum(A)) - A;
n = size(A,1);
I = eye(n);

[risk0,~,~] = para4Measure(m, L); % risk for measure m at for the original network

% actual conflict for three different internal opinion vectors
realRisk0 = zeros(1,3); 
[~, ~, MReal0] = para4Measure(m, L);
s1= randn(n,1); % s1 - the random one consists of -1 and 1
for is=1:n
    if s1(is) > 0
        s1(is) = 1;
    else
        s1(is) = -1;
    end
end
[lvec,lval] = eig(L);
% s2 - corresponds to the 10th samllest eigenvalue; low-frequency
s2 = sign(lvec(:,10));
% s3 - corresponds to the 10th largest eigenvalue; high-frequency
s3 = sign(lvec(:,n-10)); 
realRisk0(1) = s1'*MReal0*s1;
realRisk0(2) = s2'*MReal0*s2;
realRisk0(3) = s3'*MReal0*s3;

%% parameters for algorithms

% upper bounds for total wieght changes that can be made at every iteration.
p = 2; % 2 for budget 1 since the A is symmetric

% stepsize for coordinate descent, usually set it to 1, but 0<=stepsize<=1
% is feasible
stepsize = 1;

% for WCR optimization
values = []; % for saving the worst case conflict values
dim = 10; % number of worst case opinions

% real risks with 3 artificially created opinion vectors
% realRisks = zeros(1,3);
% MM = inv(L+I)^2*L;MM=(MM+MM')/2;
for k = 1:1
    if avgCase
        % for average s, optimise tr(M) over L
        
        % find the current worst case risk when optimising the avg case risk
%         [~, ~, M] = para4Measure(m, L);
%         cvx_begin
%             variable X(n,n) symmetric
%             maximize sum(sum(X.*M))
%             subject to
%                 diag(X) == 1
%                 X == semidefinite(n)
%         cvx_end
%         C = chol(X)';
%         V = sign(C*randn(size(C,2),dim)); % relaxation
%         values = [values; cvx_optval max(diag(V'*M*V))];
        [~, Gm, ~] = para4Measure(m, L);
        if gradient
            % avg + gradient
            cvx_begin
                variable step(n,n) symmetric
                maximize sum(sum((diag(sum(step)) - step).*Gm))
                subject to 
                    diag(step) == zeros(n,1)
                    A-1*step >= 0
                    A-1*step <= 1
                    sum(sum(abs(step))) <= p % change one edge at one iteration
%                     sum(sum(abs(step))) <= (n^2 - n)/2
%                     sum(sum(step.^2)) <= 0.99^k
            cvx_end
            if sum(sum(round(A - step, 5))) == 0
                break;
            end
            A = A - 1*step;
        else
            % avg + coordinate
            Incr = 1000*ones(size(A));
            Decr = 1000*ones(size(A));
            for i=1:1:n
                for j = i+1:1:n
                    delta = Gm(i,i) + Gm(j,j) - Gm(i,j) - Gm(j,i);
                    Decr(i,j) = -delta;
                    Incr(i,j) = delta;
                    if A(i,j) > 1-stepsize
                        Incr(i,j) = 1000;
                    elseif A(i,j) < stepsize
                        Decr(i,j) = 1000;
                    end
                end
            end
            if min(min(Decr)) < min(min(Incr)) && min(min(Decr)) <= 0
                [imin, jmin] = find(Decr == min(min(Decr)));
                if size(imin,1) > 1
                    imin = imin(1);jmin = jmin(1);
                end
                A(imin, jmin) = A(imin, jmin) - stepsize;
                A(jmin, imin) = A(imin, jmin);
            elseif min(min(Incr)) <= 0 || round(min(min(Incr)),15) == 0
                [imin, jmin] = find(Incr == min(min(Incr)));
                if size(imin,1) > 1
                    imin = imin(1);jmin = jmin(1);
                end
                A(imin, jmin) = A(imin, jmin) + stepsize;
                A(jmin, imin) = A(imin, jmin);
            else
                break;
            end
        end
        L = diag(sum(A)) - A;
        [risksAVG(k), ~, ~] = para4Measure(m, L);
        toc
    else
        [~, ~, M] = para4Measure(m, L); % get the middle matrix for measure m
        % for worst case s, optimise the worst case tr(s'*M*s) over L
        % First solve maxcut problem to find worst-case binary opinion vector, 
        % or a set of approximately worst-case binary opinion vectors:
        cvx_begin
            variable X(n,n) symmetric
            maximize sum(sum(X.*M))
            subject to
                diag(X) == 1
                X == semidefinite(n)
        cvx_end
    
        C = chol(X)';
        V = sign(C*randn(size(C,2),dim)); % relaxation
    
        values = [values; cvx_optval max(diag(V'*M*V))];
        
        iLI2 = inv(L+I)^2;
        
        if gradient
            % worst + gradient
            % iLI2*V*V'*iLI2 - M*V*V'*M is the gradient of
            % v*v'*inv(L+I)^2*L, only for the external conflict
            Gwm = worstCaseRiskGradient(m, L, V)
%             Gwm = iLI2*V*V'*iLI2 - M*V*V'*M;
            cvx_begin
                variable step(n,n) symmetric
                maximize sum(sum(Gwm.*(diag(sum(step))-step)))
                subject to
                    diag(step) == zeros(n,1)
                    A-1*step >= 0
                    A-1*step <= 1
                    sum(sum(abs(step))) <= p
            cvx_end
            A = A-1*step;
        else
            % worst + greedy
            XX = worstCaseRiskGradient(m, L, V);
%             XX = iLI2*V*V'*iLI2 - M*V*V'*M;
            for k1=1:n
                for k2=1:n
                    XXnew(k1,k2) = XX(k1,k1)+XX(k2,k2)-XX(k1,k2)-XX(k2,k1);
                end
            end
            XX = XXnew;
            clear XXnew
            for k1=1:n
                for k2=1:n
                    % check 0<=A+step<=1
                    if XX(k1,k2)>0 & A(k1,k2)==0
                        XX(k1,k2)=0;
                    elseif XX(k1,k2)<0 & A(k1,k2)==1
                        XX(k1,k2)=0;
                    end
                end
            end

            [ii,jj]=find(abs(triu(XX))==max(max(abs(triu(XX)))))
            if XX(ii(1),jj(1))>0
                disp('delete')
                A(ii(1),jj(1)) = 0;
                A(jj(1),ii(1)) = 0;
            else
                disp('add')
                A(ii(1),jj(1)) = 1;
                A(jj(1),ii(1)) = 1;
            end
        end
        L = diag(sum(A)) - A;
        [risksWST(k), ~, ~] = para4Measure(m, L);
    end
%     toc
    % real risks
%     [~, ~, MReal] = para4Measure(m, L);
%     s1= randn(n,1);
%     for is=1:n
%         if s1(is) > 0
%             s1(is) = 1;
%         else
%             s1(is) = -1;
%         end
%     end
%     [lvec,lval] = eig(L);
%     s2 = sign(lvec(:,10)); % 10th samllest eigenvalue
%     s3 = sign(lvec(:,n-10)); % 10th largest eigenvalue
%     realRisks(k,1) = s1'*MReal*s1;
%     realRisks(k,2) = s2'*MReal*s2;
%     realRisks(k,3) = s3'*MReal*s3;
%     imagesc(A),colorbar
%     pause(0.01)
end


