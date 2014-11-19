%% load data.
rating = load('ratings_data 2.txt');
[item_count,item_id] = hist(rating(:,2), unique(rating(:,2)));
[item_max, item_i] = max(item_count);
item_maxi = item_id(item_i);

%% extract feature.
feat_user = rating(rating(:,2) == item_maxi, 1);
feat = rating(rating(:,2) == item_maxi, 3);
N = length(feat_user);
user_index = zeros(max(feat_user), 1);
for i = 1:length(feat_user)
    user_index(feat_user(i)) = i;
end
feat = feat-mean(feat);

%% load trust data.
adj_matrix = zeros(N, N);
trust = load('trust_data 2.txt');
for useri = 1:length(feat_user)
    user = feat_user(useri);
    adj_matrix(useri, ...
            user_index(intersect(trust(trust(:,1) == user,2), feat_user))) = 1;
    clear user;
end
adj_matrix = adj_matrix';

%% diffusion wavelet.
addpath('../../Sparse/graph_toolbox/');
addpath('../../Sparse/');
T = directed_heat_diffusion(adj_matrix);
P = 30;
H = @(t)((eye(size(T))+t/P*T)^P);
Wres = DWPTree(H(1),3,1e-8);
W = [];
for i = 1:size(Wres,1)
    W = [W full(Wres{i,1}.ExtBasis)];
end

%% compressive sensing.
code = omp(feat, W, size(W,2), 70, 0.1); code = code';