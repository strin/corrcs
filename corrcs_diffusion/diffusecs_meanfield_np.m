function [reX mu mse mse_history] = diffusecs_meanfield_np(B, mask, X, V, S, t0, iteration)
    %% parameters.
    beta0               = 1e4;

    %% initialization.
    num_dim         = size(B, 2);
    num_node        = size(X, 2);
    mu              = zeros(num_dim, num_node);
    signal_corr     = cell(num_dim, 1);             % correlation sufficient statistics. 
    y               = cell(num_node, 1);
    beta            = ones(num_node, 1)*beta0;
    a_beta          = 1;                            % Gamma(a,b) prior on beta.
    b_beta          = 0.1;                          % Gamma(a,b) prior on beta.
    lambda          = ones(num_dim, 1);
    U               = cell(num_dim, 1);             % feature correlation matrix.
    Lambda          = zeros(num_dim,1);             % temporary feature correlation for some node.
    belief          = zeros(num_dim,1);             % mean belief propagated.
    var             = zeros(num_dim, num_node);     % marginalized variance for each entry.
    D               = rand(num_dim, num_node);
    MBBM            = cell(num_node, 1);
    v               = 1;
    K               = V*exp(-t0*S)*V';
    invK            = inv(K);
    col_run         = 2;
    row_run         = 2;

    
    for ni = 1:num_node
        y{ni} = X(mask(:,ni), ni);
        MBBM{ni}  = B(mask(:,ni),:)'*B(mask(:,ni),:);
    end
    
    for ki = 1:num_dim
        U{ki} = invK'*diag(rand(num_node, 1))*invK;
    end
    
    
    %% recovery.
    mse_old = inf;
    mse_history = [];
    for iter = 1:iteration
        
        
        % E-step. Recover latent signals. 
        for rolr = 1:row_run
            for ni = randsample(1:num_node, num_node)
                ms = B(mask(:,ni),:);
                for colr = 1:col_run
                    for ki = randsample(1:num_dim, num_dim)
                        var(ki,ni) = 1./(beta(ni)*MBBM{ni}(ki,ki)+U{ki}(ni,ni));
                        mu(ki,ni) = var(ki,ni)*(beta(ni)*ms(:, ki)'*(y{ni}-ms*mu(:,ni)+ms(:,ki)*mu(ki,ni))...
                                            -U{ki}(ni,:)*mu(ki,:)'+U{ki}(ni,ni)*mu(ki,ni));
                    end
                end
            end
        end
        
        for ki = 1:num_dim
            signal_corr{ki} = (diag(var(ki,:))+mu(ki,:)'*mu(ki,:));
        end
        
        
        % M-step. optimize hierarchical parameters.
        for ki = 1:num_dim
            % strategy 3 Laplace BCS evidence procedure.
            gamma = -1/2/lambda(ki)+sqrt(1/4/lambda(ki)^2+diag(invK*signal_corr{ki}*invK')./lambda(ki));
            D(ki,:) = 1./gamma';
%             lambda(ki) = (num_dim-1+v/2)/(sum(gamma/2)+v/2);
            U{ki} = invK'*diag(D(ki,:))*invK;
        end
        
        % M-step. update t. 
        options = optimset('GradObj','on');
        t0 = fminunc(@(t)(t_obj(t, signal_corr, V, S, D)), t0, options);
        K  = V*exp(-t0*S)*V';
        invK = inv(K);
        t0
        
        reX = B*mu;
        mse = mean(mean((X-reX).^2));
        mse_history = [mse_history; mse];
%         plot([reX(mask(:,1),1) X(mask(:,1),1)]); drawnow;
%         plot(mu(:,1)); drawnow;
        
        fprintf('iter = %f, mse = %f\n', iter, mse);
        if mse_old-mse < (1e-4)*mse
            break;
        else
            mse_old = mse;
        end
    end
end

function [f, df] = t_obj(t, signal_corr, V, S, D)
    f = 0;
    df = 0;
    alpha = 10^10;
    for ki = 1:length(signal_corr)
        f = f+trace(V*exp(-t*S)*V'*diag(D(ki,:))*V*exp(-t*S)*V'*signal_corr{ki});
        df = df-trace(V*exp(-t*S)*S*V'*diag(D(ki,:))*V*exp(-t*S)*V'*signal_corr{ki})-trace(V*exp(-t*S)*V'*diag(D(ki,:))*V*S*exp(-t*S)*V'*signal_corr{ki});
    end
    f = f+alpha*t^2;
    df = f+alpha*t;
end