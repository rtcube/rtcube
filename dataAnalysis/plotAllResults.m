
clear;
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

s1 = scatter(x2, y2, 'MarkerFaceColor',[0.9 0.38 0], 'MarkerEdgeColor',[0.9 0.38 0]);
s2 = scatter(x3, y3, 'MarkerFaceColor',[1 0.72 0.38], 'MarkerEdgeColor',[1 0.72 0.38]);
s3 = scatter(x5, y5, 'MarkerFaceColor',[0.69 0.67 0.82], 'MarkerEdgeColor',[0.69 0.67 0.82]);
s4 = scatter(x7, y7, 'MarkerFaceColor',[0.37 0.23 0.6], 'MarkerEdgeColor',[0.37 0.23 0.6]);

l = legend(['8 nodes best fit (slope coefficient = ' num2str(coeffs1(1), 2) ')' ], ...
           ['16 nodes best fit (slope coefficient = ' num2str(coeffs2(1), 2) ')' ], ... 
           ['24 nodes best fit (slope coefficient = ' num2str(coeffs3(1), 2) ')' ], ...
           ['32 nodes best fit (slope coefficient = ' num2str(coeffs4(1), 2) ')' ], ...
            '8 nodes', '16 nodes', '24 nodes', '32 nodes', 'Location','northwest');
set(l, 'FontSize', 18);
set(l, 'Color','none');
set(l, 'EdgeColor', 'none');
set(gca,'FontSize',16)

hold off;

figure;
subplot(1, 2, 1);
plot(X(:, 1), X(:, 2), 'LineWidth', 3);
xlabel('Number of nodes');
set(gca,'FontSize',16)

p = subplot(1, 2, 2);
hold on;

plot(X(:, 1), X(:, 3), X(:, 1), X(:, 4), X(:, 1), X(:, 5), 'LineWidth', 3);
set(p,'FontSize',16,'YTick',...
    [0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500]);
legend('Mean', 'Max', 'Min');

xlabel('Number of nodes');
ylabel('Time [ms]');
set(gca,'FontSize',16)

hold off;

