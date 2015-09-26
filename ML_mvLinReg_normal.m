%% ML Course (STANFORD) - MV. Linear Regression (Normal Equation)
%% Joan Cardona, 26/09/2015

%% Implementation of multivariate linear regression using the normal eq.

%% The first part of the implementation is very similar to the one
%% we used before, using Gradient Descent.

data = load('ex1data2.txt'); %load the data into a new variable

data_dimensions = size(data); %makes a length 2 vector [47 3]

m = data_dimensions(1); % 47 training set examples

n = data_dimensions(2) - 1; % 3 variables - 1 dependent variable y

X = [ones(m,1), data(:,1:2)]; % x_1: all ones for convenience.

y = data(:,3);

%% Having these variables, we can compute theta very quickly, just
%% using:

theta = pinv(X'*X)*X'*y;

%% No need to scale, no need to normalize.

predict1 = [1 (1600) (3)] * theta;
fprintf('For 1600 squared feet and 3 bedrooms we predict a price of %f\n',...
    predict1);

predict1 = [1 (2000) (3)] * theta;
fprintf('For 2000 squared feet and 3 bedrooms we predict a price of %f\n',...
    predict1);

%% It's even a little bit better than Gradient Descent.

%% Just note that the Normal Equation method can be very slow
%% if the number of features n is very large, since the computation
%% of the inverse of a matrix is a difficult task.