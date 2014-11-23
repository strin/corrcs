function x_hat = bp(y, a, n, eps)
    f = ones(2 * n, 1);
    A = -1 * eye(2 * n);
    b = zeros(2 * n, 1);
    Aeq = [a+eps (-1 * a)+eps];
    x = linprog(f, A, b, Aeq, y);
    x_hat = x(1 : n, 1) - x(n + 1 : 2 * n, 1);