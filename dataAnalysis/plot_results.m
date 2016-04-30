function plot_results(filename)
	X = csvread(filename);
	X = X(100:end, :);
	figure()
	plot(X(:, 1), X(:, 2) - X(:, 3));
	disp(filename);
	title(filename);
end
