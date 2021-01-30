function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(X,2);
hypo = sigmoid(X*theta);
% You need to return the following variables correctly 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i = 1:m
    J = J + (y(i).*log(hypo(i))+(1-y(i)).*log(1-hypo(i)));
    for j = 1:n
        grad(j) = grad(j) + (hypo(i) - y(i)).*X(i,j);
    end
end
J = (-1./m).*J + lambda.*(theta.'*theta - (theta(1).*(theta(1))))./(2*m);
grad = grad.*(1./m);
    grad(j) = grad(j) + lambda.*theta(j)./m;
end