function [ A, x, y ] = loadResults( quererFile, generatorFile, nodes )

Q = csvread(quererFile);
G = csvread(generatorFile);

X = ones(600, 3);
Y = G(2:601,:);

for i = size(Q, 1):-1:1
    if Q(i,1) == 0
        break;
    end
    X(Q(i,1),:) = Q(i,:); 
end

% X = X(100:500,:);
% Y = Y(100:500,:);

y = X(:,3) - Y(:,3);
x = X(:,1);
    
coefs = polyfit(X(:,1), y, 1);
A = [nodes, coefs(1), mean(y), max(y), min(y)];

end

