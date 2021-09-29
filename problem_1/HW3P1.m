fsz = 16;

% x = 0:5;
% x = x .* pi ./ 10;
% x = x(:, 3:6);
% y = x .^ 2;
% m11 = 1/12 * sum(y);
% m12 = -1/12 * sum(x);
% m21 = m12;
% m22 = 1/3;
% M = [m11, m12; m21, m22];
% %% Question 1. plots
nt = 100;
t = linspace(-1,1,nt);
[am,bm] = meshgrid(t,t);

sol = zeros(nt,nt);
gas = zeros(nt,nt);
gbs = zeros(nt,nt);

for i = 1 : nt
   for j = 1 : nt
       sol(i,j) = f(am(i,j), bm(i,j));
       [ga,gb] = gf(am(i,j), bm(i,j));
       gas(i,j) = ga;
       gbs(i,j) = gb;
   end
end
gas = gas.^2;
gbs = gbs.^2;


figure(1);clf;
surf(t,t,sol);
% contourf(t,t,sol,linspace(min(min(sol)),max(max(sol)),20));
% colorbar;
set(gca,'Fontsize',fsz);
xlabel('a','Fontsize',fsz);
ylabel('b','Fontsize',fsz);

gs = (gas + gbs) .^ 1/2;
figure(2);clf;
surf(t,t,gs);
set(gca,'Fontsize',fsz);
xlabel('a','Fontsize',fsz);
ylabel('b','Fontsize',fsz);

%% Question 2. gradient descent
% the minimal stepsize of going to the flat region
[ga1,gb1] = gf(1,0);
stepflat = (pi/2) / (pi/2*ga1 - gb1); % 1.5100

maxiter = 5000;
err_grad = 1;
%plot various stepsizes
figure(3);clf;
hold on;
for stepsize = 0.1 %here we can insert 1.5100, 1.4949 etc to finish part (b) of the probelm
    fs = zeros(1, maxiter);
    a = 1;
    b = 0;
    iter = 1;
    while err_grad > 1e-8 && iter < maxiter
        fs(iter) = f(a,b);
        [ga,gb] = gf(a,b);
        err_grad = (ga^2 + gb^2) ^ 1/2;
        a = a - stepsize * ga;
        b = b - stepsize * gb;
        iter = iter+1;
    end
    plot(1:maxiter, fs, 'DisplayName', sprintf('stepsize: %.2f', stepsize));
    legend('-DynamicLegend');
    legend('show');
end

set(gca,'Fontsize',fsz);
xlabel('iteration','Fontsize',fsz);
ylabel('f(a,b)','Fontsize',fsz);

fprintf("The GD solution is a=%d, b=%d, f(a,b)=%d\n", a, b, f(a,b));
fprintf("when stepsize is %d, the iteration number is %d", stepsize, iter);

% x = 0:5;
% x = x.*pi./10;
% test = a*x - b

% %% Question 3. stochastic gradient descent
a = 1;
b = 0;
maxiter = 500;
fs = zeros(1, maxiter);
iter = 1;
k = 1;
stepsize = 1;
while k < 10
    kstep = ceil (2 ^ k / k);
    stepsize_k = 2 ^ (-k) * stepsize;
    for i = 1:kstep
        fs(iter) = f(a,b);
        [ga,gb] = sgf(a,b);
        a = a - stepsize * ga;
        b = b - stepsize * gb;
        iter = iter + 1;
    end
    k = k + 1;
end


figure(4);clf;
hold on;
plot(1:maxiter, fs)
set(gca,'Fontsize',fsz);
xlabel('iteration','Fontsize',fsz);
ylabel('f(a,b)','Fontsize',fsz);
fprintf("The SGD solution is a=%d, b=%d, f(a,b)=%d\n", a, b, f(a,b));

%% functions
function [ga, gb] = sgf(a, b)
    x = 0:5;
    x = x.*pi./10;
    i = randi(6);
    rel = relu(a*x(i)-b);
    reg = rel - g(x(i));
    grel = grelu(a*x(i)-b);
    ga = x(i)*grel*reg / 6;
    gb = - grel*reg / 6;
end

function [ga, gb] = gf(a, b)
    x = 0:5;
    x = x.*pi./10;
    ga = 0;
    gb = 0;
    for i = 1:6
        rel = relu(a*x(i)-b);
        reg = rel - g(x(i));
        grel = grelu(a*x(i)-b);
        ga = ga + x(i)*grel*reg;
        gb = gb + grel*reg;
    end
    ga = ga / 6;
    gb = - gb / 6;
end

function res = f(a, b)
    x = 0:5;
    x = x.*pi./10;
    res = 0;
    for i = 1:6
       re = relu(a*x(i)-b) - g(x(i));
       res = res + re^2;
    end
    res = res / 12;
end

function re = relu(x)
    re = max(x,0);
end

function gre = grelu(x)
    if x > 0
        gre = 1;
    else
        gre = 0;
    end
end

function gx = g(x)
    gx = 1 - cos(x);
end

function [a,b,flag] = solvef(i)
    x = i:5;
    x = x.*pi./10;
    gx = 1 - cos(x);
    x2 = x.^2;
    temp = sum(gx)/(6-i);
    a_n = sum(x.*gx) - sum(x)*temp;
    a_d = sum(x2) - 1/(6-i) * sum(x)^2;
    a = a_n / a_d;
    b = (a * sum(x) - sum(gx)) / (6-i);
    
    res = f(a,b);
    
    flag = 1;
    if a * x(1) - b < 0
        flag = 0;
    end
    
    if i > 0 && a * (i-1)*pi/10 - b > 0
        flag = 0;
    end
    
end