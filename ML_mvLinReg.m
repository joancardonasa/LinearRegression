%% Machine Learning Course (STANFORD) - Multivariate Linear Regression 
%% Joan Cardona, 26/09/2015

%% Implementation of multivariate (n > 1) linear regression using G.D.

%% The data, in this case, ex1data2.txt contains a training set of housing
% prices in Portland, Oregon. 
% The first column (x_1) is the size of the house (in square feet);
% The second column is the number of bedrooms (x_2 will be discrete); 
% The third column is the price of the house (y, dependant variable).

%% In other words, we want to know what a good market price would be given
% the size of the house and the number of bedrooms.

data = load('ex1data2.txt'); %load the data into a new variable

data_dimensions = size(data); %makes a length 2 vector [47 3]

m = data_dimensions(1); % 47 training set examples

n = data_dimensions(2) - 1; % 3 variables - 1 dependent variable y

%% Let us define the variables X & y:

X = [ones(m,1), data(:,1:2)]; % x_1: all ones for convenience.

y = data(:,3);

figure;
scatter3(X(:,2), X(:,3), y, 'filled') % plot a 3-D scatterplot of data
xlabel('Size of the house in feet^2'); % Set the x_1-axis label
ylabel('Number of rooms'); % Set the x_2-axis label
zlabel('Price of the house'); % Set the y label
% No need to plot x_0 because it only consists of values of 1 and
% it is defined for convenience of computing the matrix.
hold off

%% Hypothesis:
%% Our hypothesis is based upon the fact that the linear model that
%% lives within the structure of the data is represented by 
%% h(x) = theta_0 * x_0 + theta_1 * x_1 + theta_2 * x_2

%% theta will be a vector of length n + 1.

%% h(x) = y; h(x) - y = 0;
%% To upgrade this estimation, let's define that the cost function J
%% that we wish to minimize tweaking the theta parameters is given by:

%% J(theta) = (1/2 * m) * sum((h(x(i)) - y(i))^2) for i = 1:m

theta = zeros(n + 1, 1);

%% In this algorithm we will also implement Feature Scaling 
%% and Mean Normalization so that the algorithm runs smoother and faster.
%% This process must be taken into account and reverted at the end when
%% we use it to predict values.

X(:,1); % No need to do it for the vector of ones

mu_2 = mean(X(:,2));
mu_3 = mean(X(:,3));

sd_2 = std(X(:,2));
sd_3 = std(X(:,3));

X(:,2) = ( X(:,2) - mu_2 ) /( sd_2 );
X(:,3) = ( X(:,3) - mu_3 ) /( sd_3 );

%% Gradient Descent settings (May have to be modified):

num_iters = 600; % Initial 1500
iterations = 1:1:num_iters; % vector containing 1 to num_iters (steps of 1)

alpha = 0.01; % Initial 0.01 learning rate

J = 0; % Cost Function J(theta)
J = (1/(2 * m)) * sum((X*theta - y).^2); 
% 1/(2*47) = 1/94 = 0.01063829787
% X * theta - y; sizes: (47 x 3)*(3 x 1) - (47 x 1) = (47 x 1)

J_history = zeros(num_iters, 1); % Let's define a vector that
                                 % stores the result of J for each
                                 % iteration.
                                 
% The next step is to implement gradient descent to minimize this value J:

for iter = 1:num_iters 
    
    t_0 = theta(1,:) - alpha * (1/m) * sum((X*theta - y) .* X(:,1));
    t_1 = theta(2,:) - alpha * (1/m) * sum((X*theta - y) .* X(:,2));
    t_3 = theta(3,:) - alpha * (1/m) * sum((X*theta - y) .* X(:,3));
 
    theta(1,:) = t_0;
    theta(2,:) = t_1;
    theta(3,:) = t_3;
    
    % The storing of the values must be done in this order so that 
    % the values of theta are all updated SIMULTANEOUSLY!!!
    
    J_history(iter) = (1/(2 * m)) * sum((X*theta - y).^2);
    
end

% X(:,2) = ( X(:,2) + (mu_2/sd_2) ) * sd_2;
% X(:,3) = ( X(:,3) + (mu_3/sd_3) ) * sd_3;

% theta(2,:) = (theta(2,:) + (mu_2/sd_2) ) * sd_2;
% theta(3,:) = (theta(3,:) + (mu_3/sd_3) ) * sd_3;

%% Plot tryout of how the model fits each one of the variables:

% Note that the variables have been scaled and normalized!!!

figure;
subplot(1,2,1);
plot(X(:,2), y, 'rx', 'MarkerSize', 10)
hold on
plot(X(:,2), X(:,1) * theta(1,:) + X(:,2) * theta(2,:))

subplot(1,2,2);
plot(X(:,3), y, 'rx', 'MarkerSize', 10);
hold on
plot(X(:,3), X(:,1) * theta(1,:) + X(:,3) * theta(3,:))

% This may sound dumb, but I seem to be unable to plot the linear model
% into the data in a 3 dimensional space.

% Update 22:48: I just understood that a three dimensional line cannot
% be plotted since the number of bedrooms is a discrete value. 

%% Let's see how the value of J decreases for each iteration
figure;
plot(iterations, J_history)
xlabel('Number of iterations of Gradient Descent'); % x-axis label
ylabel('Value of J(theta)'); % y-axis label

%% A couple of Prediction Examples:

predict1 = [1 (1600 - mu_2)/(sd_2) (3 - mu_3)/(sd_3)] * theta;
fprintf('For 1600 squared feet and 3 bedrooms we predict a price of %f\n',...
    predict1);

predict1 = [1 (2000 - mu_2)/(sd_2) (3 - mu_3)/(sd_3)] * theta;
fprintf('For 2000 squared feet and 3 bedrooms we predict a price of %f\n',...
    predict1);




