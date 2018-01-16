function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% n = length(X(:,1));

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

	% disp(n);

	Y1=-y;
	Y2=-(Y1+1);

	P=X';
	T=theta';

	S=T*P;

	HX1=sigmoid(S);
	HX2=(1-HX1);

	Sum=Y1'*log(HX1')+Y2'*log(HX2');

	J=Sum/m;

	% disp(theta);

	T=theta.^2;

	% disp(theta);

	Sum=sum(T);
	Sum=Sum-(theta(1,1)^2);

	Sum=(lambda/(2*m))*Sum;

	% disp(theta(1,1));

	J=J+Sum;

	n=size(X,2);

	X0=X(:,1);
	X1=X(:,2:n);

	% disp(X0(1:5));
	% disp(X1(1:5));

	T=HX1'-y;

	% disp(size(T));
	% disp(size(X0));

	T1=T'*X0;
	T1/=m;

	% disp(T1/m);

	T2=T'*X1;
	T2=T2/m;

	% disp(size(T2'));

	T2=T2'+(lambda/m)*theta(2:length(theta),:);

	grad=[T1;T2];

% =============================================================

end
