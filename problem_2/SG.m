function [fall,norg] = SG(nt,N,tol,iter_max)
fsz = 16; % fontsize
%% setup training mesh
t = linspace(0,1,nt+2);
[xm,ym] = meshgrid(t,t);
I = 2:(nt+1);
xaux = xm(I,I);
yaux = ym(I,I);
xy = [xaux(:),yaux(:)]';
Ntrain = nt*nt; % the number of training points
%% initial guess for parameters
N = 10; % the number of hidden nodes (neurons)
npar = 4*N; % the total number of parameters
w = ones(npar,1); % initially all parameters are set to 1
%%
[r,J] = Res_and_Jac(w,xy);
f = F(r);
g = J'*r/Ntrain;
nor = norm(g);
fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
%% The trust region BFGS method
tic % start measuring the CPU time
iter = 0;
k = 1;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
norg(1) = nor;
fall(1) = f;
% TODO: DEFINE STEPSIZE 
% alpha = 1;
% above is learning rate that I chose
% Define size of batch for random choice of indices
batch_size = 5;
batch_ind = zeros(batch_size);
% above is what I set for batch size
while k < 1000 % && nor > tol
    % TODO: insert the stochastic gradient descend algorithm here 
    % below is what I add
%     kstep = ceil (2 ^ k / k);
%     alpha_k = 2 ^ (-k) * alpha;
    alpha_k = k ^ -1;
%     for ii = 1:kstep
        w = w - alpha_k * g;
        [r,J] = Res_and_Jac(w,xy);
%     x_index = randi(nt);
%     y_index = randi(nt);
%     x_cur = xaux(1, x_index);
%     y_cur = yaux(y_index, 1);
%     xy = [x_cur, y_cur]';
        f = F(r);
        for i = 1:batch_size
            batch_ind(i) = randi(Ntrain);
        end
        g = 0;
        for i = 1:batch_size
            g = g + (J(batch_ind(i), :)' * r(batch_ind(i)));
        end
        g = g / batch_size;
        nor = norm(g);
    % above is what I add     
        fprintf('iter %d: f = %d, norg = %d\n',iter,f,nor);
        iter = iter + 1;
        norg(iter+1) = nor;
        fall(iter+1) = f;
%     end
    k = k + 1;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter,f,nor);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
%% visualize the solution
nt = 101;
t = linspace(0,1,nt);
[xm,ym] = meshgrid(t,t);
[fun,~,~,~] = ActivationFun();
[v,W,u] = param(w);
[f0,f1,g0,g1,~,~,~,~,h,~,~,~,exact_sol] = setup();
A = @(x,y)(1-x).*f0(y) + x.*f1(y) + (1-y).*(g0(x)-((1-x)*f0(0)+x*f1(0))) + ...
     y.*(g1(x)-((1-x)*f0(1)+x*f1(1)));
B = h(xm).*h(ym);
NNfun = zeros(nt);
for i = 1 : nt
    for j = 1 : nt
        x = [xm(i,j);ym(i,j)];
        NNfun(i,j) = v'*fun(W*x + u);
    end
end
sol = A(xm,ym) + B.*NNfun;
esol = exact_sol(xm,ym);
err = sol - esol;
fprintf('max|err| = %d, L2 err = %d\n',max(max(abs(err))),norm(err(:)));

%
figure(1);clf;
contourf(t,t,sol,linspace(min(min(sol)),max(max(sol)),20));
colorbar;
set(gca,'Fontsize',fsz);
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);

%
figure(2);clf;
contourf(t,t,err,linspace(min(min(err)),max(max(err)),20));
colorbar;
set(gca,'Fontsize',fsz);
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);
%
figure(3);clf;
subplot(2,1,1);
fall(iter+2:end) = [];
plot((1:iter+1)',fall,'Linewidth',2,'Marker','.','Markersize',20);
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
subplot(2,1,2);
norg(iter+2:end) = [];
plot((1:iter+1)',norg,'Linewidth',2,'Marker','.','Markersize',20);
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('|| grad f||','Fontsize',fsz);
end

%% the objective function
function f = F(r)
    f = 0.5*r'*r/length(r);
end
