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

	% disp(size(y));
	% disp(size(sigmoid(X)));

	% feed-forward

	a1=X;
	a1=[ones(m,1) a1]; 	% disp(size(a1));
	z2=a1*Theta1'; 		% disp(size(z2));
	a2=sigmoid(z2);		% disp(size(a2)); 
	a2=[ones(m,1) a2];	% disp(size(a2));
	z3=a2*Theta2';		% disp(size(Theta2));
	a3=sigmoid(z3);		% disp(size(a3));

	% a3 = h(x)
	% Unregularized cost

	for k=1:num_labels
		yk=y==k; 		% logical array
		htemp=a3(:,k);	% htheta(x)_k
		% disp(size(htemp));
		Jk=-yk'*log(htemp)-(1-yk)'*log(1-htemp);
		J=J+Jk;
	end

	J=J/m;

	t1=Theta1.^2;
	t2=Theta2.^2;

	% disp(size(t1));
	% disp(size(t2));

	% Regularized

	extra=sum(sum(t1(:,2:end)))+sum(sum(t2(:,2:end)));
	extra=extra*lambda/(2*m);

	J=J+extra;
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

	% fprintf('Theta2 size: %d\n', size(Theta2));

	htheta=a3;

	for t=1:m
		a3=htheta(t,:);
		a3=a3';

		yk=zeros(size(a3));

		for k=1:num_labels
			if(y(t)==k)
				yk(k,:)=1;
			end
		end

		delta3=(a3-yk);

		% disp(size(delta3));
		delta2=Theta2'*delta3.*sigmoidGradient([1, z2(t,:)])';
		% disp(size(temp));
		delta2=delta2(2:end);

		Theta1_grad=Theta1_grad+delta2*a1(t,:);
		Theta2_grad=Theta2_grad+delta3*a2(t,:);

	end

	% disp(size(Theta1_grad));
	% disp(size(Theta2_grad));

	Theta1_grad=Theta1_grad/m;
	Theta2_grad=Theta2_grad/m;
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
	
	% Regularized


	T1=(lambda/m)*Theta1(:,2:end);
	T2=(lambda/m)*Theta2(:,2:end);

	Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+T1;
	Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+T2;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
