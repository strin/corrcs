function [Z score res] = beta_ising(phi, y, adj_list, W, iteration, feat)
    %% parameters.
    a = 1; b = 1;
    sigma2 = 0.1; % measurement noise.
    siginv = 0.5*1/sigma2; % division is expensive!
    
    %% initialization.
    n = size(y,2);
    dim = size(phi,2);
    Z = zeros(dim,n);  % infered pr[z_ik = 1]
    Yp = phi'*y;
    Wt = W';
    WWt = W+Wt;
    PtP = phi'*phi;
    dPtP = diag(PtP);
    
    %% algorithm.
    last_res = inf;
    for iter = 1:iteration
        fprintf('iter = %d \r', iter);

        % update pi.
        for k = 1:dim
            if iter == 1
                pia = ones(dim,1)*a/dim;
                pib = ones(dim,1)*b*(1-1/dim);
                break;
            else    
                pia(k) = a/dim+sum(Z(k,:));
                pib(k) = b*(1-1/dim)+n-sum(Z(k,:));
            end
        end
        piapib = psi(pia)-psi(pib);

        % update Z.
        for i = 1:n
            zp = PtP*Z(:,i)-dPtP.*Z(:,i);
            if iter > 1 && isempty(adj_list{i}) == 0
                net = sum(WWt*(2*Z(:, adj_list{i})-1),2);
            else
                net = zeros(dim,1);
            end
            % add correlation term.
            for k = 1:dim
                z1 = piapib(k)-siginv*(PtP(k,k)-2*(Yp(k,i)-PtP(k,:)*Z(:,i)+PtP(k,k)*Z(k,i)))+2*net(k);
                Z(k,i) = exp(z1)./(exp(z1)+1);
                if isinf(exp(z1))
                    Z(k,i) = 1;
                end
            end
        end
        
        res = norm(abs(phi*Z-phi*feat), 'fro');
        score = fscore(Z, feat);
        if abs(res-last_res)/abs(res) < 10^(-4)
            break
        else
            last_res = res;
        end
        fprintf('GCS \t f1 = %f, residual = %f \r', score, res);
    end
    fprintf('\n');
end