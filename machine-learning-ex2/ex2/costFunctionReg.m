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

theta_0 = theta(1);
theta_rest = theta(2:end);

X_0 = X(:, 1);
X_rest = X(:, 2:end);

g = sigmoid(X * theta);
term_1 = -y .* log(g);
term_2 = (1 - y) .* log(1 - g);
term_3 = (lambda / (2 * m)) * sum((theta_rest .^ 2));
J = ((1 / m) * sum(term_1 - term_2)) + term_3;

grad_0 = (1 / m) * sum((g - y) .* X_0)

grad_rest_term_1 = ((1 / m) * sum((g - y) .* X_rest));
grad_rest_term_2 = ((lambda / m) .* theta_rest');

grad_rest = grad_rest_term_1 + grad_rest_term_2;

grad = [grad_0 grad_rest];



% =============================================================

end
