function [reX code] = bcs(B, mask, X)
    %% parameters.
    beta0           = 1e4;
    param.iteration = 30;
    param.diag      = 0;
    param.pgsample  = 10;
    param.c         = 1;
    
    %% initialization.
    num_dim         = size(B, 2);
    num_node        = size(X, 2);
    signal_mu       = cell(num_node, 1);
    signal_sigma    = cell(num_node, 1);
    signal_corr     = cell(num_node, 1);
    y               = cell(num_node, 1);
    alpha           = cell(num_node, 1);
    beta            = ones(num_node, 1)*beta0;
    lambda          = ones(num_node, 1);
    MBBM            = cell(num_node, 1);
    mask            = logical(mask);
    v               = 1;
    
    
    for ni = 1:num_node
        y{ni} = X(mask(:,ni), ni);
        alpha{ni} = rand(num_dim, 1);
        MBBM{ni}  = B(mask(:,ni),:)'*B(mask(:,ni),:);
        signal_sigma{ni} = inv(beta(ni)*MBBM{ni}+diag(alpha{ni}));
        signal_mu{ni} = beta(ni)*signal_sigma{ni}*B(mask(:,ni),:)'*y{ni};
        if param.diag
            signal_corr{ni} = diag(signal_sigma{ni}+signal_mu{ni}*signal_mu{ni}');
        else
            signal_corr{ni} = signal_sigma{ni}+signal_mu{ni}*signal_mu{ni}';
        end
    end
    %% recovery.
    for iter = 1:param.iteration
%         iter        
        % M-step. optimize hierarchical parameters.
        for ni = 1:num_node
            % strategy 3 Laplace BCS evidence procedure.
            gamma = -1/2/lambda(ni)+sqrt(1/4/lambda(ni)^2+(signal_mu{ni}.*signal_mu{ni}+diag(signal_sigma{ni}))/lambda(ni));
            beta(ni) = (num_dim+2)/(norm(y{ni}-B(mask(:,ni),:)*signal_mu{ni})^2+(num_dim-sum(diag(signal_sigma{ni}).*alpha{ni}))/beta(ni));
            alpha{ni} = 1./gamma;
            lambda(ni) = (num_dim-1+v/2)/(sum(gamma/2)+v/2);
            
%             % strategy 1 BCS E-M.
%             beta(ni) = length(y{ni})/(norm(y{ni}-B(mask(:,ni),:)*signal_mu{ni})^2 ...
%                             +(num_dim-diag(signal_sigma{ni})'*alpha{ni})/beta(ni));
%             alpha{ni} = 1./(signal_mu{ni}.*signal_mu{ni}+diag(signal_sigma{ni}));

%             % strategy 2 BCS evidence procedure.
%               gamma = (1-alpha{ni}.*diag(signal_sigma{ni}));
%               alpha{ni} = gamma./(signal_mu{ni}.^2);
%               beta(ni)  = (length(y{ni})-sum(gamma))/norm(y{ni}-B(mask(:,ni),:)*signal_mu{ni})^2;
        end
        
        % E-step. estimate code.
        for ni = 1:num_node
            signal_sigma{ni} = inv(beta(ni)*MBBM{ni}+diag(alpha{ni}));
            signal_mu{ni} = beta(ni)*signal_sigma{ni}*B(mask(:,ni),:)'*y{ni};
        end
        
        code = reshape(cell2mat(signal_mu), num_dim, num_node);
        reX = B*code;
        
        mse = mean(mean((X-reX).^2));
        fprintf('iter = %f, mse = %f\n', iter, mse);
%         % V-step. visualization. 
%         i = 1;
%         subplot(1,2,1);
%         x = X(:,i);
%         plot(x);  hold on;
%         x(mask(:,i) == 0) = NaN;
%         plot(x,'r.');
%         subplot(1,2,2);
%         plot(B*signal_mu{1}); drawnow; 
%         beta(1)
        
%         alpha{ni}
%         code = reshape(cell2mat(signal_mu), num_dim, num_node);
%         reX = B*code;
%         mse = mean(mean((X-reX).^2));
%         fprintf('graph compressed sensing: err = %f\n', mse);
%         beta(1)
    end
    
    
%     mse = mean(mean((X-reX).^2));
%     fprintf('graph compressed sensing: err = %f\n', mse);
    
end