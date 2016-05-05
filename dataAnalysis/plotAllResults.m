
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

s1 = scatter(x2, y2, 'c', 'filled');
s2 = scatter(x3, y3, 'b', 'filled');
s3 = scatter(x5, y5, 'g', 'filled');
s4 = scatter(x7, y7, 'o', 'filled');
legend([s1, s2, s3, s4], '4 nodes', '8 nodes', '16 nodes', '32 nodes');

xlabel('Iteration');
ylabel('Time [ms]');
xlim([100 500]);
ylim([0 1200]);

coeffs = polyfit(x2, y2, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs, fittedX);
plot(fittedX, fittedY, 'r-', 'LineWidth', 2);
t = text('FontWeight','bold','FontSize',14, 'String',['Coef = ' num2str(coeffs(1))],...
    'Position',[fittedX(100)-10 fittedY(100)+10], 'Color',[1 0 0]);
set(t, 'Rotation', 14);

coeffs = polyfit(x3, y3, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs, fittedX);
plot(fittedX, fittedY, 'r-', 'LineWidth', 2);
t = text('FontWeight','bold','FontSize',14, 'String',['Coef = ' num2str(coeffs(1))],...
    'Position',[fittedX(100)-10 fittedY(100)+10], 'Color',[1 0 0]);
set(t, 'Rotation', 12);

coeffs = polyfit(x5, y5, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs, fittedX);
plot(fittedX, fittedY, 'r-', 'LineWidth', 2);
t = text('FontWeight','bold','FontSize',14, 'String',['Coef = ' num2str(coeffs(1))],...
    'Position',[fittedX(100)-10 fittedY(100)+10], 'Color',[1 0 0]);
set(t, 'Rotation', 11);

coeffs = polyfit(x7, y7, 1);
fittedX = linspace(100, 500, 200);
fittedY = polyval(coeffs, fittedX);
plot(fittedX, fittedY, 'r-', 'LineWidth', 2);
t = text('FontWeight','bold','FontSize',14, 'String',['Coef = ' num2str(coeffs(1))],...
    'Position',[fittedX(100)-10 fittedY(100)+10], 'Color',[1 0 0]);
set(t, 'Rotation', 9);

hold off;



% figure;
% X(9,:) = plot_results('results36.txt', 36, 1, 2, 1);
% X(10,:) = plot_results('results40.txt', 40, 1, 2, 2);

figure;
subplot(2, 3, 1);
plot(X(:, 1), X(:, 3));
title('Linear regression');
xlabel('Number of nodes');

subplot(2, 3, 2);
plot(X(:, 1), X(:, 2));
title('Mean');
xlabel('Number of nodes');
ylabel('Time [ms]');

subplot(2, 3, 3);
plot(X(:, 1), X(:, 4));
title('Max');
xlabel('Number of nodes');
ylabel('Time [ms]');

subplot(2, 3, 4);
plot(X(:, 1), X(:, 5));
title('STD');
xlabel('Number of nodes');
ylabel('Time [ms]');

subplot(2, 3, 5);
plot(X(:, 1), X(:, 6));
title('Median');
xlabel('Number of nodes');
ylabel('Time [ms]');

subplot(2, 3, 6);
plot(X(:, 1), X(:, 7));
title('Variance');
xlabel('Number of nodes');

% figure;
% X(9,:) = plot_results('results36.txt', 36, 1, 2, 1);
% X(10,:) = plot_results('results40.txt', 40, 1, 2, 2);

figure;
plotBandwidth('results4.txt', 4, 2, 2, 4);
plotBandwidth('results8.txt', 8, 2, 2, 1);
plotBandwidth('results16.txt', 16, 2, 2, 2);
plotBandwidth('results20.txt', 20, 2, 2, 4);
plotBandwidth('results24.txt', 24, 2, 2, 3);
plotBandwidth('results28.txt', 28, 2, 2, 4);
plotBandwidth('results32.txt', 32, 2, 2, 4);

