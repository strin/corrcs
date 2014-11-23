%% parameters. 
dim = 500;

%% load data.
% station = csvread('pm25/station.csv');
pollution = load('pollution.mat');
feat = pollution.feat.pm25;
n = pollution.n;
feat = feat-repmat(mean(feat), size(feat,1), 1);
feat = feat./repmat(std(feat), size(feat, 1), 1);

%% learn basis.
dicX = im2col(feat, [dim 1]);
N = size(dicX,2);
dicX = dicX(:, randsample(1:size(dicX,2), size(dicX,2), true));
clear param; param.mode = 3; param.lambda = 100; 
param.K = dim; param.batchsize = 512; param.iter = ceil(size(dicX,2)/param.batchsize);
D = mexTrainDL(dicX, param);
K = size(D,2);
alpha = zeros(K, N);
for i = 1:N
    i
    alpha(:,i) = omp(dicX(:,i), D, size(D,2), param.lambda, 1)';
end
save('bayes_diffusion.mat');

%% build graph from station.
load('bayes_diffusion.mat');
station = csvread('pm25/station.csv');
n = size(station, 1);

x = station(:,2);
y = station(:,3);
E = size(n,n);
lambda = 10;
for i = 1:n
    for j = 1:n
        if i ~= j
            E(i,j) = lambda*exp(-lambda*sqrt((x(i)-x(j))^2+(y(i)-y(j))^2));
        end
    end
end
H = E;
for i = 1:n
    H(i,i) = -sum(H(i,:));
end
[U S] = eig(H);
K = @(t)(U*diag(exp(diag(t*S)))*U');
Z = alpha;
