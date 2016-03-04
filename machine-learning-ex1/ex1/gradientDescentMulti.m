function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

   TH = zeros(size(theta));
   for i = 1:size(theta)
      TH(i) = theta(i, 1);
   end

   hyp = X * theta;
   % sigma_diff = sum(hyp - y)
   % alpha_sigma = (alpha / m) * sum((hyp - y)' * X)

   for i = 1:size(theta)
      T(i) = TH(i) - ((alpha / m) .* sum((hyp - y)' * X(:, i)));
   end

   for i = 1:size(theta)
      TH(i) = T(i);
   end

   theta = TH;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
