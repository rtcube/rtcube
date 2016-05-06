function [X, x, y] = loadResults(filename, nodes)

	X = csvread(filename);
	X = X(100:end, :);
    
    %y = (X(:, 2) - X(:, 3) + abs(min( X(:, 2) - X(:, 3))));
    %600 = max[ abs(min( X(:, 2) - X(:, 3))) ] po wszystkich eksperymentach
    
    y = (X(:, 2) - X(:, 3) + 600);
    x = X(:,1);
    
    coefs = polyfit(X(:,1), y, 1);
    
    X = [nodes, coefs(1), mean(y), max(y), min(y)];
end
