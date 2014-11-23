%%% Bayesian Compressive Sensing
% y: measurement, B: basis, prec: precision of measurement.
function x = bcs(y, B, prec)
    BTB = B'*B;
    m = size(B,2);
    n = size(B,1);
    alpha = rand(m, 1);
    for iter = 1:1000
        sigma = pinv(prec*BTB+diag(alpha));
        mu = prec*sigma*B'*y;
        
        prec = n/(norm(y-B*mu)^2+(m-sum(diag(sigma).*alpha))/prec);
        alpha = 1./(mu.*mu+diag(sigma));
        
%         gamma = 1-alpha.*diag(sigma);
%         alpha = gamma./mu.^2;
%         prec = (n-sum(gamma))/(norm(y-B*mu, 2)^2);
    end
    x = mu;
end