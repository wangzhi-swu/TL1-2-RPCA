function [X, E]=RPCA(D)
% time_DCA=tic;

%% initialization
[m,n] = size(D);

L = D;
norm_two = lansvd(L, 1, 'L');
norm_inf = norm( L(:), inf) ;
dual_norm = max(norm_two, norm_inf);
L = zeros( m, n);
converged = false;
theta = .3/ sqrt(max(m,n));
mu = .08/norm_two*sqrt(m);
mu_bar = mu*1e7;
rho = 1.05;          % this one can be tuned
par.alpha = 1;
par.theta = theta;
tol = 1e-7;
i = 0;
maxIter = 1000;


X = zeros( m, n);
E = zeros( m, n);
while ~converged
    i = i+1;
   %% compute E
    temp_T = D - X + (1/mu)*L;
    E = max(temp_T - theta/mu, 0);
    E = E+min(temp_T + theta/mu, 0);
    
   %% compute X
    par.mu = mu;
    X =truncatedL1L2(D, E, L, par);
   
    L = L+mu*(D-X-E);
    mu = min(mu*rho, mu_bar);
    %% 
    stopCriterion = norm(D-X-E, 'fro') / norm(D, 'fro');
    if stopCriterion < tol
        converged = true;
    end
    if ~converged && i >= maxIter
        
%         disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
%     if mod( i, 10) == 0
%         disp(['iter ' num2str(i) ' r(A) ' num2str(rank(X))...
%             ' |E|_0 ' num2str(length(find(abs(E)>0)))...
%             ' stopCriterion ' num2str(stopCriterion)]);
%     end    
end
% time_DCA=toc(time_DCA);
end
