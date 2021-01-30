function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp_theta = theta;
temp_theta(1) = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
err = X*theta - y;
J = ((err.'*err)+lambda.*(sum(theta(2:end).^2)))./(2.*m);
reg_term = lambda.*temp_theta./m;
grad = (1./m).*(X'*err) + reg_term;
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
