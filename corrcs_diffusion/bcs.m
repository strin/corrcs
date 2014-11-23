function [reX code] = bcs(B, mask, X)
    %% parameters.
    beta0           = 1e4;
    param.iteration = 1000;
    param.diag      = 0;
    param.pgsample  = 10;
    param.c         = 1;
    
    %% initialization.
    num_dim         = size(B, 2);
    num_node        = size(X, 2);
    mu              = zeros(num_dim, num_node);
    prec            = cell(num_node, 1);
    sigma           = cell(num_node, 1);
    y               = cell(num_node, 1);
    my              = cell(num_node, 1);
    code            = zeros(num_dim, num_node);
    alpha           = cell(num_node, 1);
    beta            = ones(num_node, 1)*beta0;
    lambda          = ones(num_node, 1);
    MBBM            = cell(num_node, 1);
    V               = cell(num_node, 1);
    V2              = cell(num_node, 1);
    VtV             = cell(num_node, 1);
    mb              = cell(num_node, 1);
    mask            = logical(mask);
    v               = 1;
    
    
    
    for ni = 1:num_node
        y{ni} = X(mask(:,ni), ni);
        alpha{ni} = rand(num_dim, 1);
        MBBM{ni}  = B(mask(:,ni),:)'*B(mask(:,ni),:);
        [V{ni}, mb{ni}] = eig(MBBM{ni});
        V2{ni} = V{ni}.^2;
        VtV{ni} = diag(V{ni}'*V{ni});
        mb{ni} = diag(mb{ni});
        sigma{ni} = 1./(beta(ni)*mb{ni}+alpha{ni});
        my{ni} = V{ni}'*B(mask(:,ni),:)'*y{ni};
        mu(:,ni) = beta(ni)*sigma{ni}.*my{ni};
    end
    
    %% recovery.
    last_lhood = -inf;
    for iter = 1:param.iteration
%         iter        
        % M-step. optimize hierarchical parameters.
        for ni = 1:num_node
            % strategy 3 Laplace BCS evidence procedure.
            gamma = -1/2/lambda(ni)+sqrt(1/4/lambda(ni)^2+(mu(:,ni).*mu(:,ni).*VtV{ni}+V2{ni}*sigma{ni})/lambda(ni));
%             beta(ni) = (num_dim+2)/(norm(y{ni}-B(mask(:,ni),:)*mu(:, ni))^2+(num_dim-sum(sigma{ni}.*VtV{ni}.*alpha{ni}))/beta(ni));
            alpha{ni} = 1./gamma;
            lambda(ni) = (num_dim-1+v/2)/(sum(gamma/2)+v/2);
        end
        
        % E-step. estimate code.
        lhood = 0;
        for ni = 1:num_node
            prec{ni} = beta(ni)*mb{ni}+alpha{ni};
            sigma{ni} = 1./prec{ni};
            mu(:,ni) = beta(ni)*sigma{ni}.*my{ni};
            code(:,ni) = V{ni}*mu(:,ni);
            lhood = lhood-1/2*(sum(mu(:,ni).*prec{ni}.*mu(:,ni)))-1/2*sum(log(sigma{ni}));
%             sigma{ni} = 1/beta(ni)*sigma{ni};
        end
        
        reX = B*code;
        
        mse = mean(mean((X-reX).^2));
        fprintf('iter = %f, mse = %f, lhood = %f\n', iter, mse, lhood);
        if lhood-last_lhood < 1e-4*abs(lhood)
            break;
        else
            last_lhood = lhood;
        end
%         % V-step. visualization. 
%         i = 1;
%         subplot(1,2,1);
%         x = X(:,i);
%         plot(x);  hold on;
%         x(mask(:,i) == 0) = NaN;
%         plot(x,'r.');
%         subplot(1,2,2);
%         plot(B*mu{1}); drawnow; 
%         beta(1)
        
%         alpha{ni}
%         code = reshape(cell2mat(mu), num_dim, num_node);
%         reX = B*code;
%         mse = mean(mean((X-reX).^2));
%         fprintf('graph compressed sensing: err = %f\n', mse);
%         beta(1)
    end
    
    
%     mse = mean(mean((X-reX).^2));
%     fprintf('graph compressed sensing: err = %f\n', mse);
    
end