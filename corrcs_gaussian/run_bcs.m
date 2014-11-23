m_ratio = 0.7;
bgraph = graph>0.9*max(abs(graph(:)));
mask = zeros(size(data));
for ni = 1:size(data,2)
    mask(randsample(1:m,ceil(m*m_ratio)),ni) = 1;
end
if trace(bgraph(1,1).^2) ~= 0
    bgraph = bgraph.*(ones(size(bgraph))-eye(size(bgraph)));
end
res = cell(m-dim+1);
nowX = cell(m-dim+1);
nowMask = cell(m-dim+1);
for j = 1:m-dim+1
    nowX{j} =  X(:, j:(m-dim+1):end);
    nowMask{j} = mask(j:j+dim-1,:);
end
parfor j = 1:m-dim+1
    j
    res{j} = bcs(B, nowMask{j}, nowX{j}, bgraph);
end
reX_g = zeros(size(X));
for j = 1:m-dim+1
    reX_g(:, j:(m-dim+1):end) = res{j};
end
clear res;
data_c_g =  zeros(size(data));
data_re_g = zeros(size(data));
for ni = 1:n
    for j = 1:m-dim+1
        data_c_g(j:j+dim-1,ni) = data_c_g(j:j+dim-1,ni)+1;
        data_re_g(j:j+dim-1,ni) = data_re_g(j:j+dim-1,ni)+reX_g(:, (ni-1)*(m-dim+1)+j);
    end
end
data_re_g = data_re_g./data_c_g;
mse = mean(mean((data-data_re_g).^2));
fprintf('graph compressed sensing: err = %f\n', mse);
save('polling_bayesian_bcs.mat');