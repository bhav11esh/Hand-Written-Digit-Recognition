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
X = [ones(size(X,1),1), X];

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
z2=X*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1) a2];
z3=a2*Theta2';
H_theta=sigmoid(z3);
temp=zeros(m,num_labels);
for i=1:m
	temp(i,y(i,1))=1;
end
for i=1:m
	for j=1:num_labels
		J=J+(1/m)*(-1*temp(i,j)*log(H_theta(i,j))-(1-temp(i,j))*log(1-H_theta(i,j)));
	end
end
reg=0;
hls1=size(Theta1,1);
ils=size(Theta1,2);
ols=size(Theta2,1);
hls2=size(Theta2,2);
for i=2:ils
	for j=1:hls1
		reg=reg+(Theta1(j,i)^2);
	end
end
for i=2:hls2
	for j=1:ols
		reg=reg+(Theta2(j,i)^2);
	end
end
J=J+reg*(lambda/(2*m));
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
Delta_1=zeros(size(Theta1));
Delta_2=zeros(size(Theta2));
for i=1:m
	Del3=(H_theta(i,:).-temp(i,:));
	Del2=(Del3*Theta2(:,2:end)).*sigmoidGradient(z2(i,:));
	Delta_2=Delta_2+Del3'*a2(i,:);
	Delta_1=Delta_1+Del2'*X(i,:);
end
Theta1_grad=Delta_1./m;
Theta2_grad=Delta_2./m;	
%delta3 = H_theta - temp;

%Theta2_grad = Theta2_grad + (delta3'*a2)/m;

%delta2 = delta3*Theta2(:,2:end).*sigmoidGradient(z2);

%Theta1_grad = Theta1_grad+(delta2'*X)/m;
	
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
reg_grad1=(lambda/m)*Theta1;
reg_grad1(:,1)=0;
reg_grad2=(lambda/m)*Theta2;
reg_grad2(:,1)=0;
Theta1_grad=Theta1_grad+reg_grad1;
Theta2_grad=Theta2_grad+reg_grad2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

