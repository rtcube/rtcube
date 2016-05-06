function [x, y] = plotBandwidth(filename)

    X = csvread(filename);
	X = X(99:end, :);
    
    for i = 2:size(X, 1)
        X(i, 2) = X(i,3)-X(i-1,3);
    end

    X = X(2:end, 1:2);
    X(:,2) = X(:,2)./1000;
    X(:,2) = X(:,2).*128.*9./1000;
    
    x = X(:,1);
    y = X(:,2);

%     ax = subplot(m, n, p);
% 	plot(X(:, 1), X(:,2));
% 	title(['Number of nodes = ' num2str(nodes)]);
% 
%     ylim(ax, [0 1]);
%     xlim(ax, [100 500]);
%     
%     xlabel('Iteration');
%     ylabel('Speed [Gigabits per second]');


end