function [reX code] = gcs(B, mask, X, graph, W)
    %% parameters.
    beta0               = 1e2;
    param.iteration     = 200;
    param.diag          = 0;
    param.pgsample      = 10;
    param.c             = 1;
    param.gaussround    = 1;
    %% initialization.
    num_dim         = size(B, 2);
    num_node        = size(graph, 1);
    signal_mu       = cell(num_node, 1);
    signal_sigma    = cell(num_node, 1);
    signal_corr     = cell(num_node, 1);
    y               = cell(num_node, 1);
    alpha           = cell(num_node, 1);
    beta            = ones(num_node, 1)*beta0;
    a_beta          = 1;                            % Gamma(a,b) prior on beta.
    b_beta          = 0.1;                          % Gamma(a,b) prior on beta.
    lambda          = ones(num_node, 1);
    MBBM            = cell(num_node, 1);
    mask            = logical(mask);
    v               = 1;
    
    [edgeI, edgeJ]  = find(graph);
    
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
        iter        
        % M-step. optimize hierarchical parameters.
        for ni = 1:num_node
            % strategy 3 Laplace BCS evidence procedure.
            gamma = -1/2/lambda(ni)+sqrt(1/4/lambda(ni)^2+(signal_mu{ni}.*signal_mu{ni}+diag(signal_sigma{ni}))/lambda(ni));
            beta(ni) = (num_dim+a_beta*2)/(norm(y{ni}-B(mask(:,ni),:)*signal_mu{ni})^2+(num_dim-sum(diag(signal_sigma{ni}).*alpha{ni}))/beta(ni)+b_beta);
            alpha{ni} = 1./gamma;
            lambda(ni) = (num_dim-1+v/2)/(sum(gamma/2)+v/2);
            
            
%             beta(ni) = length(y{ni})/(norm(y{ni}-B(mask(:,ni),:)*signal_mu{ni})^2 ...
%                             +(num_dim-diag(signal_sigma{ni})'*alpha{ni})/beta(ni));
%             alpha{ni} = 1./(signal_mu{ni}.*signal_mu{ni}+diag(signal_sigma{ni}));

%             % strategy 2 evidence procedure.
%               gamma = (1-alpha{ni}.*diag(signal_sigma{ni}));
%               alpha{ni} = gamma./(signal_mu{ni}.^2);
%               beta(ni)  = (length(y{ni})-sum(gamma))/norm(y{ni}-B(mask(:,ni),:)*signal_mu{ni})^2;
        end
        
        % E-step. compute signal.
        for gi = 1:param.gaussround
            for ni = 1:num_node
                if param.diag
                    pp_sigma = zeros(num_dim, 1);
                else
                    pp_sigma = zeros(num_dim, num_dim);
                end
                pp_mu = zeros(num_dim, 1);
                for nj = 1:num_node
                    if graph(ni, nj) == 0
                        continue;
                    end
                    pp_sigma = pp_sigma+W;
                    pp_mu = pp_mu+W*signal_mu{nj};
                end    
                if param.diag
                    pp_sigma = diag(pp_sigma);
                end
                signal_sigma{ni} = inv(beta(ni)*MBBM{ni}+diag(alpha{ni})+pp_sigma);
                signal_mu{ni} = signal_sigma{ni}*(beta(ni)*B(mask(:,ni),:)'*y{ni}+pp_mu);
                if param.diag
                    signal_corr{ni} = diag(signal_sigma{ni}+signal_mu{ni}*signal_mu{ni}');
                else
                    signal_corr{ni} = signal_sigma{ni}+signal_mu{ni}*signal_mu{ni}';
                end
            end
        end
%         for ni = 1:1
%             signal_sigma{ni} = pinv(beta(ni)*MBBM{ni}+diag(alpha{ni}));
%             signal_mu{ni} = beta(ni)*signal_sigma{ni}*B(mask(:,ni),:)'*y{ni};
%         end
        
        % V-step. visualization. 
%         i = 1;
%         subplot(1,2,1);
%         x = X(:,i);
%         plot(x);  hold on;
%         x(mask(:,i) == 0) = NaN;
%         plot(x,'r.');
%         subplot(1,2,2);
%         plot(B*signal_mu{1}); drawnow; 
%         beta(1)
        
        code = reshape(cell2mat(signal_mu), num_dim, num_node);
        subplot(1,3,1); 
        plot(B*code(:,1)); drawnow;
        subplot(1,3,2);
        plot(B*code(:,2)); drawnow;
        subplot(1,3,3);
        plot(X(:,1));
        
%         mse = sum(sum(mask.*(X-reX).^2))./sum(mask(:));
%         plot(mean((X-reX).^2)); drawnow;
    end
    code = reshape(cell2mat(signal_mu), num_dim, num_node);
    reX = B*code;
    mse = mean(mean((X-reX).^2));
    
end