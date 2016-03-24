function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 -------------------------------------------------------------

% Add ones to the X data matrix
A1 = [ones(m, 1) X];

% size(X)
% size(A1)

Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

%sz_A2 = size(A2)

% Add bias term to the A2 data matrix
A2 = [ones(m, 1) A2];
Z3 = A2 * Theta2';
hyp = sigmoid(Z3);

sz_hyp = size(hyp)


% make a binary vector of y-values: 2 => [0; 1; 0; 0; ...0], 5 => [0; 0; 0; 0; 1; 0; ...0]
K = num_labels;
for i = 1 : m
   for k = 1 : K
      ybin = zeros(K, 1);
      if y(i) == k 
         ybin(k, 1) = 1;
      endif

      %sz_ybin = size(ybin)
      %ybin'

      %hyp(i, :)
      %log(hyp(i, :))

      term_1 = ybin(k, 1) * log(hyp(i, k));
      term_2 = (1 - ybin(k, 1)) * log(1 - hyp(i, k));
      J = J .+ sum(term_1 + term_2);

   endfor % k
endfor % i

J = -(1 / m) * J;

% Need for regularization -------------------------------------------
   Theta1_rest = Theta1(:, 2:end);
   Theta2_rest = Theta2(:, 2:end);

if lambda > 0
   size(Theta1_rest)
   size(Theta2_rest)

   Theta1_reg_term = 0;
   for j = 1 : hidden_layer_size
      for k = 1 : input_layer_size
         Theta1_reg_term = Theta1_reg_term + Theta1_rest(j, k) ^ 2;
      endfor % k
   endfor % j
   Theta1_reg_term

   Theta2_reg_term = 0;
   for j = 1 : num_labels
      for k = 1 : hidden_layer_size
         Theta2_reg_term = Theta2_reg_term + Theta2_rest(j, k) ^ 2;
      endfor % k
   endfor % j
   Theta2_reg_term

   reg_term = (lambda / (2 * m)) * (Theta1_reg_term + Theta2_reg_term)

   J = J + reg_term
endif % lambda > 0

% Part 2 -------------------------------------------------------------
% Back propagation
printf("Back propagation...\n");

delta_3 = zeros(m, K);

for i = 1 : m
   ybin = zeros(K, 1);
   for k = 1 : K
      if y(i) == k 
         % y(i)
         ybin(k, 1) = 1;
      endif

      %sz_ybin = size(ybin)
      %ybin

      %hyp(i, :)

   endfor % k
   %ybin'
   %hyp(i, :)

   delta_3(i, :) = hyp(i, :) .- ybin';
endfor % i


%sz_Theta2_rest_transpose = size(Theta2_rest')
%sz_delta_3 = size(delta_3)
%sz_sigmoidGradient_Z2 = size(sigmoidGradient(Z2))

delta_2 = (delta_3 * Theta2_rest) .* sigmoidGradient(Z2);

%sz_delta_2 = size(delta_2)

delta_2 = delta_2(2:end);

A2 = sigmoid(Z2); % reinit as A2 was polluted earlier with adding ones'
Delta_2 = zeros(hidden_layer_size, num_labels);
Delta_2 = Delta_2 + (A2' * delta_3);
%sz_Delta_2 = size(Delta_2)


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
