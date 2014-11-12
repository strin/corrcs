%% build graph from station.
load('bayes_diffusion.mat');
station = csvread('pm25/station.csv');
n = size(station, 1);
Z = alpha;

%% parameters.
iteration = 50;
num_run = 1;
mratio_list = [0.05:0.05:1];
% mratio_list = [0.5];
param.mode = 0; param.L = 20;

%% inference.
result = zeros(length(mratio_list), num_run);
mse_history = cell(length(mratio_list), num_run);
mu = cell(length(mratio_list), num_run);
reX = cell(length(mratio_list), num_run);

for run = 1:num_run
    for i = 1:length(mratio_list)
        i
        mratio = mratio_list(i);
        num_dim0 = size(feat,1);
        skip = num_dim0-dim+1;
        offset = run;
        subZ = Z(:, offset:skip:N);
        subX = dicX(:, offset:skip:N);   
        mask = rand(dim, n) < mratio;
        mu{i,run} = zeros(size(D,2), n);
        param.lambda = 1e-3/(size(D,1)*mratio);
        for iter = 1:30
            param.L = iter;
            for ni = 1:n
    %             mu{i}(:,ni) = mexLasso(subX(mask(:,ni),ni), D(mask(:,ni),:), param);
    %              mu{i}(:,ni) = omp(subX(mask(:,ni),ni), D(mask(:,ni),:), size(D,2), 40, 1)';
                mu{i}(:,ni) = mexOMP(subX(mask(:,ni),ni), D(mask(:,ni),:), param);
            end
            reX{i,run} = D*mu{i};
            mse = mean(mean((reX{i,run}-subX).^2));
            mse_history{i, run} = mse;
            result(i,run) = mse;
            fprintf('mratio = %f, mse = %f\n', mratio, mse);
        end
%         mu{i} = mexLasso(subX(mask(:,ni),:), D(mask(:,ni),:), param);
    end
end
clear Z;
clear dicX;
clear alpha;
clear pollution;
% % save('../result/result_bp_iid_multirun.mat');
save('../result/result_omp_iid_converge.mat');
