
clear;
X(1,:) = plot_results('results4.txt', 4);
X(2,:) = plot_results('results8.txt', 8);
X(3,:) = plot_results('results12.txt', 12);
X(4,:) = plot_results('results16.txt', 16);
X(5,:) = plot_results('results20.txt', 20);
X(6,:) = plot_results('results24.txt', 24);
X(7,:) = plot_results('results28.txt', 28);
X(8,:) = plot_results('results32.txt', 32);
X(9,:) = plot_results('results36.txt', 36);
X(10,:) = plot_results('results40.txt', 40);

figure;
plot(X(:, 1), X(:, 3));
title('Slope');

figure;
plot(X(:, 1), X(:, 2));
title('Mean');

figure;
plot(X(:, 1), X(:, 4));
title('Max');

figure;
plot(X(:, 1), X(:, 5));
title('STD');

figure;
plot(X(:, 1), X(:, 6));
title('Median');

