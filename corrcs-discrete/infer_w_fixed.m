%% load data.
load('epinion_binary_rate(mean).mat');

%% 
% find fix W using Maximum Liklihood of tr(sum(z_i W z_j))+lambda_0 ||W||_F^2.
% lambda0 = 1;
% W = zeros(dim,dim);
% count = 0;
% for i = 1:n
%     if isempty(adj_list{i}) == 1
%         continue
%     end
%     for j = adj_list{i}'
%         W = W+1/2/lambda0*feat(:,i)*feat(:,j)';
%         count = count+1;
%     end
% end
% W = W/count;
% Wt = W';
% WWt = W+Wt;
% save('epinion_partial.mat');

%% inference.
% parameter.
iteration = 1;

mratio = [0.05:0.1:1];
weight = [0 0.1 0.3 0.5 0.8 1 1.5 2];

Z = cell(length(weight), length(mratio));
score = cell(length(weight), length(mratio));
for wi = 1:length(weight)
    parfor mi = 1:length(mratio)   % measurement ratio.
        wi
        mi
        m = ceil(dim*mratio(mi));
        W = eye(dim)*weight(wi);
        phi = randn(m,dim);
        y = phi*feat;
        [Z{wi,mi}, score{wi,mi}] = beta_ising(phi, y, adj_list, W, iteration, feat);
    end
end
save('result_infer_w_fixed.mat');