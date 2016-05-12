

clear;
% [X(1,:) x2 y2] = loadResults('4_1_querer.txt', '4_1_generator.txt', 4);
% [X(2,:) x3 y3] = loadResults('6_1_querer.txt', '6_1_generator.txt', 6);
% [X(3,:) x4 y4] = loadResults('8_1_querer.txt', '8_1_generator.txt', 8);
% [X(4,:) x5 y5] = loadResults('10_1_querer.txt', '10_1_generator.txt', 10);

[X(1,:) x1 y1] = loadResults('2_1_querer.txt', '2_1_generator.txt', 2);
[X(2,:) x2 y2] = loadResults('4_1_querer.txt', '4_1_generator.txt', 4);
[X(3,:) x3 y3] = loadResults('6_1_querer.txt', '6_1_generator.txt', 6);
[X(4,:) x4 y4] = loadResults('8_1_querer.txt', '8_1_generator.txt', 8);
[X(5,:) x5 y5] = loadResults('10_1_querer.txt', '10_1_generator.txt', 10);
[X(6,:) x6 y6] = loadResults('12_1_querer.txt', '12_1_generator.txt', 12);
[X(7,:) x7 y7] = loadResults('14_1_querer.txt', '14_1_generator.txt', 14);

%Ploting results from experiments
figure;
hold on;

xlabel('Iteration');
ylabel('Time [ms]');
% xlim([100 500]);
% ylim([0 1200]);

plotPoints(x1, y1);
plotPoints(x2, y2);
plotPoints(x3, y3);
plotPoints(x4, y4);
plotPoints(x5, y5);
plotPoints(x6, y6);
plotPoints(x7, y7);

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
