function X = plot_results(filename, nodes, m, n, p)

	X = csvread(filename);
	X = X(100:end, :);
    
    %y = (X(:, 2) - X(:, 3) + abs(min( X(:, 2) - X(:, 3))));
    %600 = max[ abs(min( X(:, 2) - X(:, 3))) ] po wszystkich eksperymentach
    
    y = (X(:, 2) - X(:, 3) + 600);
    
    ax = subplot(m, n, p);
	plot(X(:, 1), y);
	title(['Number of nodes = ' num2str(nodes)]);
    
    ylim(ax, [0 1200]);
    xlim(ax, [100 500]);
    
    xlabel('Iteration');
    ylabel('Time [ms]');
    
    X = [nodes, mean(y), X(:,1)\y, max(y), std(y), median(y), var(y)];
end
