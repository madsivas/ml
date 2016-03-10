function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

M = [X y]
P = M(find(M(:,3) == 1), :);
N = M(find(M(:,3) == 0), :);
P = P(:, [1, 2]);
N = N(:, [1, 2]);
size(P)
size(N)

%plot(N, 'go;Not Admitted;', 'MarkerSize', 5); % Plot the negative data
%plot(P, 'r+rAdmitted;', 'MarkerSize', 5); % Plot the positive data

plot(N(:, 1), N(:, 2), 'go;Not Admitted;', P(:, 1), P(:, 2), 'r+;Admitted;', 'MarkerSize', 5);

%for i = 1 : size(X, 1)
%	if y(i) == 0
%		plot(X(i, 1), X(i, 2), 'go', 'MarkerSize', 5); % Plot the negative data
%	else
%		plot(X(i, 1), X(i, 2), 'r+', 'MarkerSize', 5); % Plot the positive data
%	endif
%endfor





% =========================================================================



hold off;

end
