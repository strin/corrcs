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

%% parameters.
iteration = 50;
mratio = 0.5;
dtime_list = [0.02 0.04 0.06 0.08 0.1 0.13 0.16 0.19 0.25 0.3];

%% inference.
result = zeros(length(dtime_list), 1);
mse_history = cell(length(dtime_list), 1);
for i = 1:length(dtime_list)
    dtime = dtime_list(i);
    
    num_dim0 = size(feat,1);
    skip = num_dim0-dim+1;
    offset = 1;
    subZ = Z(:, offset:skip:N);
    subX = dicX(:, offset:skip:N);   

    mask = rand(dim, n) < mratio;
    [reX mu mse mseh] = diffusecs_meanfield(D, mask, subX, K(dtime), iteration); 
    mse_history{i} = mseh;
    
    result(i) = mse;
    fprintf('i = %f, mse = %f\n', i, mse);
end