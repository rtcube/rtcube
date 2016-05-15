

clear;
% [X(1,:) x2 y2] = loadResults('4_1_querer.txt', '4_1_generator.txt', 4);
% [X(2,:) x3 y3] = loadResults('6_1_querer.txt', '6_1_generator.txt', 6);
% [X(3,:) x4 y4] = loadResults('8_1_querer.txt', '8_1_generator.txt', 8);
% [X(4,:) x5 y5] = loadResults('10_1_querer.txt', '10_1_generator.txt', 10);

[X(1,:) x1 y1 Z1 C1] = loadResults('2_1_querer.txt', '2_1_generator.txt', 2);
[X(2,:) x2 y2 Z2 C2] = loadResults('4_1_querer.txt', '4_1_generator.txt', 4);
[X(3,:) x3 y3 Z3 C3] = loadResults('6_1_querer.txt', '6_1_generator.txt', 6);
[X(4,:) x4 y4 Z4 C4] = loadResults('8_1_querer.txt', '8_1_generator.txt', 8);
[X(5,:) x5 y5 Z5 C5] = loadResults('10_1_querer.txt', '10_1_generator.txt', 10);
[X(6,:) x6 y6 Z6 C6] = loadResults('12_1_querer.txt', '12_1_generator.txt', 12);
[X(7,:) x7 y7 Z7 C7] = loadResults('14_1_querer.txt', '14_1_generator.txt', 14);

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

figure
hold on;
scatter(Z1(:,1), Z1(:,2), 'filled');
scatter(Z2(:,1), Z2(:,2), 'filled');
scatter(Z3(:,1), Z3(:,2), 'filled');
scatter(Z4(:,1), Z4(:,2), 'filled');
scatter(Z5(:,1), Z5(:,2), 'filled');
scatter(Z6(:,1), Z6(:,2), 'filled');
scatter(Z7(:,1), Z7(:,2), 'filled');
hold off;
xlabel('Time [seconds]');
ylabel('Speed [Gigabytes per second]');

figure
hold on;
scatter(C1(:,1), C1(:,2), 'filled');
scatter(C2(:,1), C2(:,2), 'filled');
scatter(C3(:,1), C3(:,2), 'filled');
scatter(C4(:,1), C4(:,2), 'filled');
scatter(C5(:,1), C5(:,2), 'filled');
scatter(C6(:,1), C6(:,2), 'filled');
scatter(C7(:,1), C7(:,2), 'filled');
hold off;
xlabel('Time [seconds]');
ylabel('Count [rows]');

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
