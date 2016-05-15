function [ A, x, y, Z, C ] = loadResults( quererFile, generatorFile, nodes )

Q = csvread(quererFile);
G = csvread(generatorFile);

X = ones(600, 3);
Y = G(2:601,:);

Z = ones(size(Q,1)-1,2);
C = ones(size(Q,1),2);

for i = 1:size(Q,1)
    C(i,1) = Q(i,3)*10^-3;
    C(i,2) = Q(i,2);
end

for i = 1:size(Q,1)-1
    Z(i,1) = Q(i+1,3)*10^-3;
    Z(i,2) = (Q(i+1,2)*16*10^-9)/((Q(i+1,3)-Q(i,3))*10^-3);
end

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

