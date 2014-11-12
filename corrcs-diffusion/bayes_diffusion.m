%% build graph from station.
load('bayes_diffusion.mat');
station = csvread('pm25/station.csv');
n = size(station, 1);

x = station(:,2);
y = station(:,3);
E = size(n,n);
lambda = 5;
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
% mratio_list = [0.05:0.05:1];
mratio_list = [0.5];
dtime = 0.1;
n_run = 5;

%% inference.
result = zeros(length(mratio_list), n_run);
mse_history = cell(length(mratio_list), n_run);

for run = 1:n_run
    for i = 1:length(mratio_list)
        mratio = mratio_list(i);
        num_dim0 = size(feat,1);
        skip = num_dim0-dim+1;
        offset = run;
        subZ = Z(:, offset:skip:N);
        subX = dicX(:, offset:skip:N);   
        mask = rand(dim, n) < mratio;

        [reX mu mse mseh] = diffusecs_meanfield(D, mask, subX, K(dtime), iteration); 
        mse_history{i, run} = mseh;
        result(i, run) = mse;
        fprintf('mratio = %f, mse = %f\n', mratio, mse);
    end
end
clear Z;
clear dicX;
clear alpha;

save('result_bayes_diffusion_multirun.mat');
