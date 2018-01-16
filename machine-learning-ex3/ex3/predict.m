function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

	disp(size(Theta1));
	disp(size(Theta2));
	disp(size(X));

	a1=[ones(m,1) X];

	disp(size(a1));

	a1=a1';

	z2=Theta1*a1;

	disp(size(z2));

	a2=sigmoid(z2);

	% disp(size(a2));

	a2=[ones(m,1)'; a2];

	disp(size(a2));

	z3=Theta2*a2;
	a3=sigmoid(z3);

	disp(size(a3));

	a3=a3';

	[pred,p]=max(a3,[],2);

% =========================================================================


end