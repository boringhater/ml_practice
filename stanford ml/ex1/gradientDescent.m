function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
params_num = size(theta,1);
temp_theta = zeros(size(theta));
for iter = 1:num_iters
    minim = X*theta - y;
    for j = 1:params_num
        for i = 1:m
            minim(i,1) = minim(i,1).*X(i,j);
        end
        features_sum = sum(minim);
        temp_theta(j,1) = theta(j,1) - (alpha/m)*features_sum;  
    end
    theta = temp_theta;
    J_history(iter) = computeCost(X, y, theta);
end
end
