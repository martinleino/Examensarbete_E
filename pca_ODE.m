%% ODE system for PCA summary statistics
% Martin Leino
% May 2023
%%
clear
close all
%% Initial conditions and parameter values
alpha = 1;          % regularization parameter
beta = 1;           % signal-to-noise ratio
c_delta = 1;
init = [0, 1];      % initial conditions
k = 2;              % order of tensor
n = 2000;
delta = c_delta/n;  % step-size
[times, m_r2_trajectory] = rungekuttasystem(

function h = PCA_ode_system(u)
R2 = @(x, y) x^2 + y;
h = [2*u(1)*(beta*k*u(1)^(k-2)-k*R2(u(1), u(2))^(k-1)-alpha/2);
    -4*u(2)*(k*R2(u(1), u(2))^(k-1) + alpha/2) +...
    c_delta*k*R2(u(1), u(2))^(k-1)];
end

