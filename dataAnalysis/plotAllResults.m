
clear;
% X(1,:) = plot_results('results4.txt', 4, 2, 4, 1);
% X(2,:) = plot_results('results8.txt', 8, 2, 4, 2);
% X(3,:) = plot_results('results12.txt', 12, 2, 4, 3);
% X(4,:) = plot_results('results16.txt', 16, 2, 4, 4);
% X(5,:) = plot_results('results20.txt', 20, 2, 4, 5);
% X(6,:) = plot_results('results24.txt', 24, 2, 4, 6);
% X(7,:) = plot_results('results28.txt', 28, 2, 4, 7);
% X(8,:) = plot_results('results32.txt', 32, 2, 4, 8);

[X(1,:) x1 y1] = loadResults('results4.txt', 4);
[X(2,:) x2 y2] = loadResults('results8.txt', 8);
[X(3,:) x3 y3] = loadResults('results16.txt', 16);
[X(4,:) x4 y4] = loadResults('results20.txt', 20);
[X(5,:) x5 y5] = loadResults('results24.txt', 24);
[X(6,:) x6 y6] = loadResults('results28.txt', 28);
[X(7,:) x7 y7] = loadResults('results32.txt', 32);

%Ploting results from experiments
figure;
hold on;

s1 = scatter(x2, y2, 'MarkerFaceColor',[0.9 0.38 0], 'MarkerEdgeColor',[0.9 0.38 0]);
s2 = scatter(x3, y3, 'MarkerFaceColor',[1 0.72 0.38], 'MarkerEdgeColor',[1 0.72 0.38]);
s3 = scatter(x5, y5, 'MarkerFaceColor',[0.69 0.67 0.82], 'MarkerEdgeColor',[0.69 0.67 0.82]);
s4 = scatter(x7, y7, 'MarkerFaceColor',[0.37 0.23 0.6], 'MarkerEdgeColor',[0.37 0.23 0.6]);

xlabel('Iteration');
ylabel('Time [ms]');
xlim([100 500]);
ylim([0 1200]);

coeffs1 = polyfit(x2, y2, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs1, fittedX);
plot(fittedX, fittedY, 'Color', [0.9 0.38 0], 'LineWidth', 2);

coeffs2 = polyfit(x3, y3, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs2, fittedX);
plot(fittedX, fittedY, 'Color', [1 0.72 0.38], 'LineWidth', 2);

coeffs3 = polyfit(x5, y5, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs3, fittedX);
plot(fittedX, fittedY, 'Color', [0.69 0.67 0.82], 'LineWidth', 2);

coeffs4 = polyfit(x7, y7, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs4, fittedX);
plot(fittedX, fittedY, 'Color', [0.37 0.23 0.6], 'LineWidth', 2);

l = legend('8 nodes', '16 nodes', '24 nodes', '32 nodes', ...
    ['8 nodes best fit (coef = ' num2str(coeffs1(1), 2) ')' ], ['16 nodes best fit (coef = ' num2str(coeffs2(1), 2) ')' ], ... 
    ['24 nodes best fit (coef = ' num2str(coeffs3(1), 2) ')' ], ['32 nodes best fit (coef = ' num2str(coeffs4(1), 2) ')' ]...
    ,'Location','northwest');
set(l, 'FontSize', 18);
set(gca,'FontSize',16)

hold off;


% figure;
% X(9,:) = plot_results('results36.txt', 36, 1, 2, 1);
% X(10,:) = plot_results('results40.txt', 40, 1, 2, 2);

% figure;
% subplot(2, 3, 1);
% plot(X(:, 1), X(:, 3));
% title('Linear regression');
% xlabel('Number of nodes');
% 
% subplot(2, 3, 2);
% plot(X(:, 1), X(:, 2));
% title('Mean');
% xlabel('Number of nodes');
% ylabel('Time [ms]');
% 
% subplot(2, 3, 3);
% plot(X(:, 1), X(:, 4));
% title('Max');
% xlabel('Number of nodes');
% ylabel('Time [ms]');
% 
% subplot(2, 3, 4);
% plot(X(:, 1), X(:, 5));
% title('STD');
% xlabel('Number of nodes');
% ylabel('Time [ms]');
% 
% subplot(2, 3, 5);
% plot(X(:, 1), X(:, 6));
% title('Median');
% xlabel('Number of nodes');
% ylabel('Time [ms]');
% 
% subplot(2, 3, 6);
% plot(X(:, 1), X(:, 7));
% title('Variance');
% xlabel('Number of nodes');

% figure;
% X(9,:) = plot_results('results36.txt', 36, 1, 2, 1);
% X(10,:) = plot_results('results40.txt', 40, 1, 2, 2);

% figure;
[x1, y1] = plotBandwidth('results4.txt');
[x2, y2] = plotBandwidth('results8.txt');
[x3, y3] = plotBandwidth('results16.txt');
[x4, y4] = plotBandwidth('results20.txt');
[x5, y5] = plotBandwidth('results24.txt');
[x6, y6] = plotBandwidth('results28.txt');
[x7, y7] = plotBandwidth('results32.txt');

