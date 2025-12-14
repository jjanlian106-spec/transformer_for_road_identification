function Td_opt = my_sqp_nmpc_fix_tyretran(Td_init, Fz, mu, omega, lambda_k, lambda_ref)

% parameters 
m  = 400;
dt = 0.001;
J  = 1.8;
r  = 0.35;

% protect against omega==0 ，lambda == 0
omega = max(omega, 1e-6);
lambda_k = max(abs(lambda_k),0);

alpha = dt/(J*omega);

% bounds
Td_max = 10000;
lb = 0*ones(4,1);
ub = Td_max*ones(4,1);


options = optimoptions('fmincon', ...
    'Algorithm','sqp', ...
    'SpecifyObjectiveGradient',false, ...
    'Display','iter', ...
    'OptimalityTolerance',1e-20);

objfun = @(Td) f_func_local(Td, omega, lambda_k, lambda_ref, Fz, mu, m, dt, r, alpha);
% 初始值是inf，这里要排查，不行求jacobi矩阵换成自己的函数(解决，初始滑移率不能是0)
%f_func_local(Td_init, omega, lambda_k, lambda_ref, Fz, mu, m, dt, r, alpha)
Td_opt = fmincon(objfun, Td_init, [], [], [], [], lb, ub,[],options)

%输出优化后结果及梯度
f_func_local(Td_opt,omega,lambda_k,lambda_ref,Fz,mu,m,dt,r,alpha)

end

function f = f_func_local(Td,omega,lambda_k,lambda_ref,Fz,mu,m,dt,r,alpha)
    sigmak = 0.15;
    J = dt/(omega*alpha);
    s_k = lambda_k*omega*r;
    vx_k = omega*r - s_k;
    tal_k = vx_k/sigmak;
    lambda_rel_k = lambda_k;
    
    [fx_k,Cx_k] = tyremodel_local(lambda_rel_k,Fz,mu);
    lambda_rel_k1 = tal_k*dt*(s_k/(omega*r)-lambda_rel_k) + lambda_rel_k;
    s_k1 = -dt*(r^2/J+1/m)*fx_k+s_k+r*dt/J*Td(1);
    [fx_k1,Cx_k1] = tyremodel_local(lambda_rel_k1,Fz,mu);
    lambda_rel_k2 = tal_k*dt*(s_k1/(omega*r)-lambda_rel_k1) + lambda_rel_k1;
    s_k2 = -dt*(r^2/J+1/m)*fx_k1+s_k1+r*dt/J*Td(2);
    [fx_k2,Cx_k2] = tyremodel_local(lambda_rel_k2,Fz,mu);
    lambda_rel_k3 = tal_k*dt*(s_k2/(omega*r)-lambda_rel_k2) + lambda_rel_k2;
    s_k3 = -dt*(r^2/J+1/m)*fx_k2+s_k2+r*dt/J*Td(3);
    [fx_k3,Cx_k3] = tyremodel_local(lambda_rel_k3,Fz,mu);
    lambda_rel_k4 = tal_k*dt*(s_k3/(omega*r)-lambda_rel_k3) + lambda_rel_k3;
    s_k4 = -dt*(r^2/J+1/m)*fx_k3+s_k3+r*dt/J*Td(4);

    r1 = lambda_ref*omega*r-s_k1;
    r2 = lambda_ref*omega*r-s_k2;
    r3 = lambda_ref*omega*r-s_k3;
    r4 = lambda_ref*omega*r-s_k4;

    %dgk1 = dgk_local(lambda_k1, r, m, alpha, dt, omega, fx_k1, Cx_k1);
    %ddgk1 = ddgk_local(r, alpha, Cx_k1);
    %dgk2 = dgk_local(lambda_k2, r, m, alpha, dt, omega, fx_k2, Cx_k2);
    %ddgk2 = ddgk_local(r, alpha, Cx_k2);
    %dgk3 = dgk_local(lambda_k3, r, m, alpha, dt, omega, fx_k3, Cx_k3);
    %ddgk3 = ddgk_local(r, alpha, Cx_k3);
    
    %计算jacobi矩阵，暂时没有用到
    % df1 = 2*r1*alpha ...
    %         -2*r2*alpha*dgk1 ...
    %         +2*r3*alpha*dgk2*dgk1 ...
    %         -2*r4*alpha*dgk3*dgk2*dgk1;
    % df2 = 2*r2*alpha ...
    %         -2*r3*alpha*dgk2 ...
    %         +2*r4*alpha*dgk3*dgk2;
    % df3 = 2*r3*alpha ...
    %         -2*r4*alpha*dgk3;
    % df4 = 2*r4*alpha;
    % df = [df1; df2; df3; df4];


    f = 1e04*r1^2+1e04*r2^2+1e04*r3^2+1e04*r4^2;
end


function [fx,Cx] = tyremodel_local(lambda_k,Fz,mu)
    B = 10; D = 1; C = 1.65;E = 0.01;
    %c1 = 0.4;
    %c2 = 33.712;
    %c3 = 0.12;
    fx = D*mu*Fz*sin(C*atan(B*lambda_k-E*(B*lambda_k-atan(B*lambda_k))));
    %fx = Fz*c1*(1-exp(-c2*lambda_k))-c3*lambda_k;
    Cx = mu * Fz * B * C * D;
    %Cx = Fz*c1*c2*exp(-c2*lambda_k)-c3;
end

function y = dgk_local(lambda_k, r, m, alpha, dt, omega, fx_k, Cx_k)
    y = r*alpha*(Cx_k*(1-lambda_k) - fx_k) + dt/(omega*m*r)*Cx_k + 1;
end

function y = ddgk_local(r, alpha, Cx_k)
    y = -2 * r * alpha * Cx_k;
end

