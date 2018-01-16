function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


	Y1=-y;
	Y2=-(Y1+1);

	P=X';
	T=theta';

	S=T*P;

	HX1=sigmoid(S);
	HX2=(1-HX1);

	Sum=Y1'*log(HX1')+Y2'*log(HX2');

	J=Sum/m;

	% disp(size(HX1));

	T=HX1'-y;

	% disp(size(T));

	T=T'*X;

	% disp(T);

	T=T/m;

	grad=T';

	% disp(J);

% =============================================================

end
