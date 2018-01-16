function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


	% disp(X);
	% disp(theta);
	% disp(theta');
	P=X';
	% disp(size(P));
	Q=theta';
	% disp(size(Q));
	R=Q*P;
	% fprintf('R: ');
	% disp(size(R));
	% fprintf('y: '); disp(size(y));
	S=R'-y;
	S=S.^2;
	% fprintf('S: '); disp(size(S));

	J=sum(S)/(2*m);


% =========================================================================

end
