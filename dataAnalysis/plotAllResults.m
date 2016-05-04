
clear;
X(1,:) = plot_results('results4.txt', 4, 2, 4, 1);
X(2,:) = plot_results('results8.txt', 8, 2, 4, 2);
X(3,:) = plot_results('results12.txt', 12, 2, 4, 3);
X(4,:) = plot_results('results16.txt', 16, 2, 4, 4);
X(5,:) = plot_results('results20.txt', 20, 2, 4, 5);
X(6,:) = plot_results('results24.txt', 24, 2, 4, 6);
X(7,:) = plot_results('results28.txt', 28, 2, 4, 7);
X(8,:) = plot_results('results32.txt', 32, 2, 4, 8);

% figure;
% X(9,:) = plot_results('results36.txt', 36, 1, 2, 1);
% X(10,:) = plot_results('results40.txt', 40, 1, 2, 2);

figure;
subplot(2, 3, 1);
plot(X(:, 1), X(:, 3));
title('Slope');
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

figure;
X(9,:) = plot_results('results36.txt', 36, 1, 2, 1);
X(10,:) = plot_results('results40.txt', 40, 1, 2, 2);

