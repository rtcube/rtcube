function X = plot_results(filename, nodes)

	X = csvread(filename);
	X = X(100:end, :);
    y = (X(:, 2) - X(:, 3) + abs(min( X(:, 2) - X(:, 3))));
    
	figure;
	plot(X(:, 1), y);
	title(filename);
    
    X = [nodes, mean(y), X(:,1)\y, max(y), std(y), median(y)];
end
