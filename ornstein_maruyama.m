%% Implementing the Euler-Maruyama method to simulate the Ornstein-Uhlenbeck process
% Martin Leino
% April 2023

%% Set the parameters of the SDE and the intial data
clear
close all
alpha = 0.5;
beta = 0.8;
X0 = 1.5;

%% Define the time interval and create a discretization
t_start = 0;
t_end = 10;
n_intervals = 1000;
dt = (t_end - t_start)/n_intervals;
t_span = t_start:dt:t_end;

%% Generate Wiener processes with mean 0 and variance dt:
n_paths = 4;
n_t_points = numel(t_span);
W = randn(n_paths, n_t_points)*sqrt(dt); % scale standard normal distribution by standard deviation
% sqrt(dt)
W = cumsum(W); % get positions as cumulative sums of increments

%% Simulate sample paths
XX = zeros(n_paths, n_t_points); % pre-allocate matrix for storing iterates of sample paths 
XX(:,1) = X0*ones(n_paths,1); % set initial data
for i = 1:n_paths
    for n = 1:n_t_points-1
        dW = W(i, n+1) - W(i, n); % increment of Wiener process
        XX(i,n+1) = XX(i,n) - alpha*XX(i,n)*dt + beta*dW;
    end
end

%% Expectation of analytical solution
X_mean = X0*exp(-alpha*t_span);

%% Plot sample paths and expectation of analytical solution
hold on
plot(t_span, XX(1,:), t_span, XX(2,:), t_span, XX(3,:), t_span, XX(4,:))
plot(t_span, X_mean, 'k*', 'MarkerSize', 3)
xlabel('Time', 'FontSize', 16)
ylabel('Value of OU-process', 'FontSize', 16)
