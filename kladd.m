clear
% syms u_2 alpha k C
% % equation_1 = 2 * k * u_2^(k-2) * (C - u_2) == alpha;
% % equation_2 = u_2 > 0;
% % equation_3 = alpha > 0;
% % equation_4 = k >= 3;
% % equation_5 = C > 0;
% % equations = [equation_1, equation_2, equation_3, equation_4, equation_5];
% % S = solve(equations, u_2, 'ReturnConditions', true);
% syms R beta k C
% eq1 = beta > 0;
% eq2 = k >= 3;
% eq3 = C > 0;
% eq4 = beta^-(2/(k-2))*R^(4*(k-1)/(k-2))+C == 0;
% eq5 = 2*R - beta^-(2/(k-2))*4*(k-1)/(k-2)*R^(4*(k-1)/(k-2)-1);
% eqs = [eq1, eq2, eq3, eq4, eq5];
% S = solve(eqs, R, 'ReturnConditions', true)
% T = solve(eqs, beta, 'ReturnConditions', true)
% % syms beta k C u_1 u_2
% % equation = u_1^2 == beta^(-2/(k-2)) * (u_1^2 + u_2)^(2*(k-1)/(k-2));
% % S = solve(equation, u_1)
syms p
test(p)

function y = test(x)
y = x^2;
end
