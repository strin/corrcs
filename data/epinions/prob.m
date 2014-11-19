%% arguments.
RATE_THRES = mean(rating(:,3));   % threshold rating into binary ones.

%% load data.
dim = 100;
rating = load('ratings_data 2.txt');
[item_count,item_id] = hist(rating(:,2), unique(rating(:,2)));
[A I] = sort(item_count);
I = I(end-dim+1:end);
% [item_max, item_i] = max(item_count);
item_maxi = item_id(I);

%% user
feat_user = [];
for i = 1:dim
    feat_user = unique([feat_user; rating(rating(:,2) == item_maxi(i), 1)]);
end
n = length(feat_user);
feat = zeros(dim, n);
for u = 1:n
    feat_u = rating(rating(:,1) == feat_user(u),2);
    feat_r = rating(rating(:,1) == feat_user(u),3);
    [res, IA, IB] =  intersect(feat_u(feat_r > RATE_THRES), item_maxi);
    feat(IB, u) = 1;
end


%% graph
user_index = zeros(max(feat_user), 1);
for i = 1:length(feat_user)
    user_index(feat_user(i)) = i;
end
trust = load('trust_data 2.txt');
adj_list = cell(n,1);
for useri = 1:n
%     useri
    user = feat_user(useri);
    adj_list{useri} = user_index(intersect(trust(trust(:,1) == user,2), feat_user));
    clear user;
end

%% analytics.
% fprintf('link percentage = %f\n', sum(sum(adj_matrix))/prod(size(adj_matrix)));
% fprintf('correlation with edges = %f\n', feat'*adj_matrix*feat/n/n);
% fprintf('correlation without edge = %f\n', sum(sum(feat*feat'))/n/n);

%% save.
save('epinion_binary_rate(mean).mat');
