%% Machine Learning Course (STANFORD) - Univariate Linear Regression 
%% Joan Cardona, 26/09/2015

%% In this exercise we will be fitting a linear model into some data.

%% The data consists of two variables:
%%  - Population of City in 10.000s' (x)
%%  - Profit in $10.000s' (y)
%% Seeing our data, we see that the training set has a linear relationship.
%% We will suppose that the variable y is linearly depenant on x.
%% Therefore, given an unknown x, we'll be able to predict y thanks
%% to the linear model we have built.

data = load('ex1data1.txt'); %load the data into a new variable "data"

%% data is a 97 x 2 matrix:

data_dimensions = size(data); %makes a length 2 vector [97 2]

m = data_dimensions(1); % 97 training set examples

n = data_dimensions(2) - 1; % 2 variables - 1 dependent variable y

%% Let us define the variables X & y:

X = [ones(m, 1), data(:,1)]; % Make a 97 x 2 matrix. Columns are:
                             % C_1: 97 1's C_2: 97 training set examples
                             % The column of ones is added for convenience
y = data(:, 2); % Make a 97 x 1 matrix or length 97 vector:
                % 97 depending variables of x

figure; % open a new figure window
plot(X(:,2), y, 'rx', 'MarkerSize', 10); % Plot the data (Scatterplot)
xlabel('Population of City in 10,000s'); % Set the x-axis label
ylabel('Profit in $10,000s'); % Set the y-axis label
    
%% Hypothesis:
%% Our hypothesis is based upon the fact that the linear model that
%% lives within the structure of the data is represented by 
%% h(x) = theta_0 + theta_1 * x

%% h(x) must be very similar to y, as to build an efficient predictor.

%% Let's establish then that:

%% h(x) = y; h(x) - y = 0;
%% To upgrade this estimation, let's define that the cost function J
%% that we wish to minimize tweaking the theta parameters is given by:

%% J(theta_0, theta_1) = (1/2 * m) * sum((h(x(i)) - y(i))^2) for i = 1:m

%% The procedure that we will implement to minimize this cost function
%% will be Gradient Descent:

theta = zeros(2, 1); % Let's define theta (0 & 1) as zeros to begin with

%% Gradient Descent settings (May have to be modified):
num_iters = 1500; % Initial 1500
iterations = 1:1:num_iters;
alpha = 0.01; % Initial 0.01

J = 0; % Cost Function J(theta)
J = (1/(2 * m)) * sum((X*theta - y).^2); 
% 1/(2*97) = 1/194 = 5.1546e-3
% X * theta - y; sizes: (97 x 2)*(2 x 1) - (97 x 1) = (97 x 1)
% We want to elevate the values in this vector to two (ELEMENT-WISE!!)
% So we add a . right before the ^ command.
% Once they have all been elevated, we sum all of the vector values
% from indexes 1 to m (All of the training examples).
% The first value of J should be 32.0727.

J_history = zeros(num_iters, 1); % Let's define a vector that
                                 % stores the result of J for each
                                 % iteration.

% The next step is to implement gradient descent to minimize this value J:

for iter = 1:num_iters 
    
    t_0 = theta(1,:) - alpha * (1/m) * sum((X*theta - y) .* X(:,1));
    t_1 = theta(2,:) - alpha * (1/m) * sum((X*theta - y) .* X(:,2));
 
    theta(1,:) = t_0;
    theta(2,:) = t_1;
    
    % The storing of the values must be done in this order so that 
    % the values of theta are all updated SIMULTANEOUSLY!!!
    
    J_history(iter) = (1/(2 * m)) * sum((X*theta - y).^2);
    
end

%% Now let's plot the linear fit:

hold on; % keep previous plot visible so we can see y and h(x)
plot(X(:,2), X*theta, '-') % X * theta (sizes: 97 x 2 * 2 x 1 = 97 x 1 
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%% A couple of Prediction Examples:

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 12] * theta;
fprintf('For population = 120,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% Let's see how the value of J decreases for each iteration
figure;
plot(iterations, J_history)
xlabel('Number of iterations of Gradient Descent'); % x-axis label
ylabel('Value of J(theta)'); % y-axis label
