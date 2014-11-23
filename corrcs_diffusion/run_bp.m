%% build graph from station.
clear;
load('bayes_diffusion.mat');
station = csvread('pm25/station.csv');
n = size(station, 1);
Z = alpha;

%% parameters.
iteration = 50;
num_run = 10;
% mratio_list = [0.05:0.05:1];
mratio_list = [0.5];
param.mode = 2; param.lambda = 1e-3;

%% inference.
result = zeros(length(mratio_list),- num_run);
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
%         param.lambda = 1/(size(D,1)*mratio);
        for ni = 1:n
%             mu{i}(:,ni) = mexLasso(subX(mask(:,ni),ni), D(mask(:,ni),:), param);
            y = subX(mask(:,ni),ni);
            dic = D(mask(:,ni),:);
            mu{i}(:,ni) = l1qc_logbarrier(dic\y, dic, dic', y, 1e-3);
%             mu{i}(:,ni) = bp(subX(mask(:,ni),ni), D(mask(:,ni),:), size(D,2), 1e-2)';
%              mu{i}(:,ni) = omp(subX(mask(:,ni),ni), D(mask(:,ni),:), size(D,2), 40, 1)';
%             mu{i}(:,ni) = mexOMP(subX(mask(:,ni),ni), D(mask(:,ni),:), param);
        end
%         mu{i} = mexLasso(subX(mask(:,ni),:), D(mask(:,ni),:), param);
        reX{i,run} = D*mu{i};
        mse = mean(mean((reX{i,run}-subX).^2));
        result(i,run) = mse;
        fprintf('mratio = %f, mse = %f\n', mratio, mse);
    end
end
clear Z;
clear dicX;
clear alpha;
clear pollution;
% % save('../result/result_bp_iid_multirun.mat');
save('../result/result_bp_iid_multirun.mat');
