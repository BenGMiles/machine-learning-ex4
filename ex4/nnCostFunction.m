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


## h(x)
# X is a matrix of our data where each row is a training example e.g
# (5000 by 400)

# X1 is the same as X but with and added row of 1s as bias
X1 = [ones(m, 1), X];

# Z1 is a matrix of X1 multiplied by the transverse of Theta1, which
# gives us our decision boundry
z1 = X1 * Theta1';
# Pass z1 to the sigmoid function to generate a matrix (a1) with all
# values between 0 and 1
a1 = sigmoid(z1);
# Add a row of 1s as bias to a1
a1 = [ones(m, 1), a1];

# Do the same as above but for Theta2. We don't need a bias row for
# this one though
z2 = a1 * Theta2';
a2 = sigmoid(z2);

## Calculate yvals
# yvals is an matrix m by num_lables that only contains 0s and 1s. With 1
# in the correct row for the correct output class in each column.
# It should be a vector, but making it a matrix makes it simpler to do
# multiplication later.

# Init an empty matrix of the right size
yvals = zeros(m, num_labels);
# For every row of data set 1 in column i
for i = 1:m
  yvals(i, y(i)) = 1;
end

## Calculate J
# Calculate initial J
J = sum(sum(-yvals .* log(a2) - (1 - yvals) .* log(1 - a2) ))/m;

# Add regularisation penalty
J = J + (sum(sum(Theta1(:,2:end) .^ 2)) +
         sum(sum(Theta2(:,2:end) .^ 2)) ) * lambda / (2 * m);

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

for t = 1:m
  % Step 1: Perform feedfoward pass
  # Set input layer values
  a_1 = X(t,:)';
  a_1 = [1; a_1];

  # Do feedfoward
  # Hidden layer
  z_2 = Theta1 * a_1;
  a_2 = sigmoid(z_2);
  a_2 = [1; a_2];

  # Ouput layer
  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);

  % Step 2: Get error for layer 3
  yt = yvals(t,:)';
  delta_3 = a_3 - yt;

  % Step 3: Get erroro for layer 2
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);

  % Step 4: Accumulate gradient, skipping delta_2(0)
  delta_2 = delta_2(2:end);

  delta_t_2 = delta_3 * a_2';
  delta_t_1 = delta_2 * a_1';

  # Add gradients for this example to Theta gradients
  Theta2_grad = Theta2_grad + delta_t_2;
  Theta1_grad = Theta1_grad + delta_t_1;
end

% Step 5: Get unregularized gradient
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
