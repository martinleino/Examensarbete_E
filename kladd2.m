% syms beta k C u_1 u_2
% eq1 = u_1^(k-2) == 1/beta*((u_1^2 + u_2)^2)^(k-1);

% Test
% n = 4, k = 3 
syms x1 x2 x3 x4
expand((x1^2 + x2^2 + x3^2 + x4^2)^2)