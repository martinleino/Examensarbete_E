%% Stability analysis
clear
syms u1 u2 alpha beta C k
% Functions in ODE system
h1 = 2*beta*k*u1^(k-1) - 2*k*u1*(u1^2+u2)^(k-1) - alpha*u1;
h2 = -4*k*u2*(u1^2+u2)^(k-1) - 2*alpha*u2 + 4*C*k*(u1^2+u2)^(k-1);
% Their derivatives
dh1_du1 = diff(h1, u1);
dh1_du2 = diff(h1, u2);
dh2_du1 = diff(h2, u1);
dh2_du2 = diff(h2, u2);

%% Case 1: k = 2, alpha = 0
disp('********** k = 2, alpha = 0 **********')
k_ = 2;
alpha_ = 0;

% Fixed point (0,0)
J1 = Jacobian(0, 0, k_, alpha_, beta, C);
disp('Eigenvalues of (0,0):')
eig(J1)

% Fixed point (0,C)
J2 = Jacobian(0, C, k_, alpha_, beta, C);
disp('Eigenvalues of (0,C):')
eig(J2)

% Fixed point (sqrt(beta-C),C)
J3 = Jacobian(sqrt(beta-C), C, k_, alpha_, beta, C);
disp('Eigenvalues of (sqrt(beta-C),C):')
eig(J3)

% Fixed point (-sqrt(beta-C),C)
J4 = Jacobian(-sqrt(beta-C), C, k_, alpha_, beta, C);
disp('Eigenvalues of (-sqrt(beta-C),C):')
eig(J4)

%% Case 2: k >= 3, alpha = 0
% Input k is a symbolic variable now
disp('********** k >= 3, alpha = 0 **********')

% Fixed point (0,0)
J5 = Jacobian(0, 0, k, alpha_, beta, C);
disp('Eigenvalues of (0,0):')
eig(J5)

% Fixed point (0,C)
J6 = Jacobian(0, C, k, alpha_, beta, C);
disp('Eigenvalues of (0,C):')
eig(J6)

%% Function giving Jacobian corresponding to the ODE system
function J = Jacobian(u1, u2 ,k, alpha, beta, C)
J = [2*beta*k*u1^(k - 2)*(k - 1) - 2*k*(u1^2 + u2)^(k - 1) - 4*k*u1^2*(u1^2 + u2)^(k - 2)*(k - 1) - alpha ...
    -2*k*u1*(u1^2 + u2)^(k - 2)*(k - 1);
    8*C*k*u1*(u1^2 + u2)^(k - 2)*(k - 1) - 8*k*u1*u2*(u1^2 + u2)^(k - 2)*(k - 1) ...
    4*C*k*(u1^2 + u2)^(k - 2)*(k - 1) - 4*k*(u1^2 + u2)^(k - 1) - 2*alpha - 4*k*u2*(u1^2 + u2)^(k - 2)*(k - 1)];
end

    

% function J = Jacobian(u_1, u_2 ,k, alpha, beta, C)
% J = [2*beta*k*(k-1)*u_1^(k-2) - 2*k*(u_1^2+u_2)^(k-2)*((u_1^2+u_2)-2*(k-1)*u_1^2)-alpha...
%     -2*k*(k-1)*u_1*(u_1^2+u_2)^(k-2);
%     8*k*(k-1)*(u_1^2+u_2)^(k-2)*(C*u_1 - u_1*u_2)...
%     4*k*(u_1^2+u_2)^(k-2)*((C-1)*(k-1)-(u_1^2+u_2)) - 2*alpha];
% end