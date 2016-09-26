for emacs:
setenv("GNUTERM","qt")
setenv("TERM","xterm")
in octave or
export TERM=xterm
export GNUTERM=qt
before octave

whos var % shows the varables dimension

https://www.gnu.org/software/octave/doc/v4.0.0/index.html
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
m = length(y); % number of training examples
h = X*theta;
J = (sum((h-y).^2) + lambda * sum(theta(2:end).^2))/2/m;
% adding 0 / zeroing out the first element in a column vector theta
grad = X'*(h-y)/m + lambda/m*[0; theta(2:end)];
% switch grad to row vector
grad = grad(:);
end
- feature normalization
function [X_norm, mu, sigma] = featureNormalize(X)
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);
end
