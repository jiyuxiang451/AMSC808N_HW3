SG(5, 10, 1e-4, 10000);
% a = 0;
% for i = 1: 16
%     a = a+ ceil(2^i/i);
% end
% 
% % nt = 5;
% % tol = 1e-4; % stop if ||J^\top r|| <= tol
% % iter_max = 10000;
% % 
% % t = linspace(0,1,nt+2);
% % [xm,ym] = meshgrid(t,t);
% % I = 2:(nt+1);
% % xaux = xm(I,I);
% % yaux = ym(I,I);
% % xy = [xaux(:),yaux(:)]';
% 
% % N = 10; % the number of hidden nodes (neurons)
% % npar = 4*N; % the total number of parameters
% % w = ones(npar,1);
% % 
% % [r,J] = Res_and_Jac(w,xy);
% % f = F(r);
% % g = J'*r/Ntrain;
% % nor = norm(g);
% % fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
% % 
% % tic % start measuring the CPU time
% % iter = 0;
% % norg = zeros(iter_max+1,0);
% % fall = zeros(iter_max+1,0);
% % norg(1) = nor;
% % fall(1) = f;
% % % TODO: DEFINE STEPSIZE 
% % while nor > tol && iter < iter_max
% %     
% %     % TODO: insert the gradient descend algorithm here 
% %     
% %     fprintf('iter %d: f = %d, norg = %d\n',iter,f,nor);
% %     iter = iter + 1;
% %     norg(iter+1) = nor;
% %     fall(iter+1) = f;
% % end
% % fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter,f,nor);
% % cputime = toc;
% % fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
% % 
% % 
% % 
% % %% the objective function
% % function f = F(r)
% %     f = 0.5*r'*r/length(r);
% % end