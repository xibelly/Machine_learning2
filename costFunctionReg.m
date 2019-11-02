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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
h = sigmoid(z);
A = -transpose(y)*log(h);
B = -transpose(1-y)*(log(1-h));
C = lambda / (2.0*m);
theta(1,1)=0;
D = transpose(theta)*theta;
J  = J + (1.0/m)*(A+B) + (C*D);
%c = lambda/m;
grad = grad + (1.0/m) * (transpose(X)*(h - y)) - ((lambda*theta)/m);




% =============================================================

end
