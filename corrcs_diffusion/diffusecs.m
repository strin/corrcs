function [reX mu mse] = diffusecs(B, mask, X, K, iteration)
    %% parameters.
    beta0               = 1e4;

    %% initialization.
    num_dim         = size(B, 2);
    num_node        = size(X, 2);
    mu              = zeros(num_dim, num_node);
    prec            = cell(num_node, 1);
    corr            = cell(num_dim, 1);             % correlation sufficient statistics. 
    y               = cell(num_node, 1);
    my              = cell(num_node);
    beta            = ones(num_node, 1)*beta0;
    lambda          = ones(num_dim, 1);
    Lambda          = ones(num_dim, 1);
    U               = cell(num_dim, 1);             % feature correlation matrix.
    var             = zeros(num_dim, num_node);     % marginalized variance for each entry.
    D               = rand(num_dim, num_node);
    MBBM            = cell(num_node, 1);
    V               = cell(num_node, 1);
    belief          = zeros(num_dim, 1);
    invK            = inv(K);
    
    for ni = 1:num_node
        y{ni} = X(mask(:,ni), ni);
        MBBM{ni}  = B(mask(:,ni),:)'*B(mask(:,ni),:);
        my{ni} = B(mask(:,ni),:)'*y{ni};
        sig = inv(beta(ni)*MBBM{ni}+diag(D(:,ni)));
        mu(:,ni) = sig*(beta(ni)*my{ni});
        var(:,ni) = diag(sig);
    end

    
    %% recovery.
    for iter = 1:iteration
        for ki = 1:num_dim
%             corr{ki} = invK2*(var(ki,:)+mu(ki,:).*mu(ki,:))';
            corr{ki} = diag(invK*(diag(var(ki,:))+mu(ki,:)'*mu(ki,:))*invK');
        end
        % M-step. optimize hierarchical parameters.
        for ki = 1:num_dim
            % strategy 3 Laplace BCS evidence procedure.
            gamma = -1/2/lambda(ki)+sqrt(1/4/lambda(ki)^2+corr{ki}/lambda(ki));
            D(ki,:) = 1./gamma';
%             lambda(ki) = (num_dim-1+v/2)/(sum(gamma/2)+v/2);
            U{ki} = invK'*diag(D(ki,:))*invK;
        end
        % E-step. Recover latent signals. 
        for ni = 1:num_node
            prec = beta(ni)*MBBM{ni};
            for ki = 1:num_dim
                Lambda(ki) = U{ki}(ni,ni);
                belief(ki) = U{ki}(ni,:)*mu(ki,:)'-U{ki}(ni,ni)*mu(ki,ni);
            end
            prec = prec+diag(Lambda);
            sig = inv(prec);
            mu(:,ni) = prec\(beta(ni)*my{ni}-belief);
            var(:,ni) = diag(sig);
        end
        reX = B*mu;
        plot([mu(:,1)]); drawnow;
        mse = mean(mean((X-reX).^2));
        fprintf('iter = %f, mse = %f\n', iter, mse);
    end
end