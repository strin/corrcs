function [reX mu mse] = diffusecs(B, mask, X, K, iteration)
    %% parameters.
    beta0               = 1e4;

    %% initialization.
    num_dim         = size(B, 2);
    num_node        = size(X, 2);
    mu              = zeros(num_dim, num_node);
    signal_sigma    = cell(num_node, 1);
    signal_corr     = cell(num_dim, 1);             % correlation sufficient statistics. 
    y               = cell(num_node, 1);
    prec            = cell(num_node, 1);
    beta            = ones(num_node, 1)*beta0;
    a_beta          = 1;                            % Gamma(a,b) prior on beta.
    b_beta          = 0.1;                          % Gamma(a,b) prior on beta.
    lambda          = ones(num_dim, 1);
    U               = cell(num_dim, 1);             % feature correlation matrix.
    Lambda          = zeros(num_dim,1);             % temporary feature correlation for some node.
    belief          = zeros(num_dim,1);             % mean belief propagated.
    var       = zeros(num_dim, num_node);     % marginalized variance for each entry.
    D               = rand(num_dim, num_node);
    MBBM            = cell(num_node, 1);
    v               = 1;
    invK            = inv(K);
    V               = cell(num_node, 1);
    V2              = cell(num_node, 1);
    VtV             = cell(num_node, 1);
    mb              = cell(num_node, 1);
    invK2           = invK.^2;
    my              = cell(num_node);
    
    for ni = 1:num_node
        y{ni} = X(mask(:,ni), ni);
        MBBM{ni}  = B(mask(:,ni),:)'*B(mask(:,ni),:);
        [V{ni}, mb{ni}] = eig(MBBM{ni});
        VtV{ni} = diag(V{ni}'*V{ni});
        V2{ni} = V{ni}.^2;
        mb{ni} = diag(mb{ni});
        prec{ni} = beta(ni)*mb{ni}+D(:,ni);
        
%         var(:,ni) = diag(inv(V{ni}*diag(prec{ni})*V{ni}'));
%         mu(:,ni) = beta(ni)*inv(V{ni}*diag(prec{ni})*V{ni}')*B(mask(:,ni),:)'*y{ni};
        
        my{ni} = V{ni}'*B(mask(:,ni),:)'*y{ni};
        var(:,ni) = V2{ni}*(1./prec{ni});
        mu(:,ni) = beta(ni)*V{ni}*(1./prec{ni}.*my{ni});
    end
    
    %% recovery.
    for iter = 1:iteration
        for ki = 1:num_dim
            signal_corr{ki} = invK2*var(ki,:)'+(invK*mu(ki,:)').^2;
        end
%         for ni = 1:num_node
%             beta(ni) = (num_dim+a_beta*2)/(norm(y{ni}-B(mask(:,ni),:)*mu(:,ni))^2+(num_dim-sum(diag(signal_sigma{ni}).*D(:,ni)))/beta(ni)+b_beta);
%         end
        % M-step. optimize hierarchical parameters.
        for ki = 1:num_dim
            % strategy 3 Laplace BCS evidence procedure.
            gamma = -1/2/lambda(ki)+sqrt(1/4/lambda(ki)^2+signal_corr{ki}/lambda(ki));
            D(ki,:) = 1./gamma';
            lambda(ki) = (num_dim-1+v/2)/(sum(gamma/2)+v/2);
            U{ki} = invK'*diag(D(ki,:))*invK;
        end
        % E-step. Recover latent signals. 
        for ni = 1:num_node
            for ki = 1:num_dim
                Lambda(ki) = U{ki}(ni,ni);
                belief(ki) = U{ki}(ni,:)*mu(ki,:)'-U{ki}(ni,ni)*mu(ki,ni);
            end
            prec{ni} = beta(ni)*V{ni}*diag(mb{ni})*V{ni}'+diag(Lambda);
            sigma = inv(prec{ni});
            mu(:,ni) = sigma*(beta(ni)*V{ni}*my{ni}-belief);
            var(:,ni) = diag(sigma);
            
            
%             for ki = 1:num_dim
%                 Lambda(ki) = U{ki}(ni,ni);
%                 belief(ki) = U{ki}(ni,:)*mu(ki,:)'-U{ki}(ni,ni)*mu(ki,ni);
%             end
%             
%             
%             signal_sigma{ni} = inv(beta(ni)*MBBM{ni}+diag(Lambda));
%             var(:,ni) = diag(signal_sigma{ni});
%             mu(:,ni) = signal_sigma{ni}*(beta(ni)*B(mask(:,ni),:)'*y{ni}-belief);
        end
        reX = B*mu;
        mse = mean(mean((X-reX).^2));
        fprintf('iter = %f, mse = %f\n', iter, mse);
    end
end