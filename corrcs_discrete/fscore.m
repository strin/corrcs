function f1 = fscore(Z, feat)
    % evaluation
    recall = sum(sum(round(Z) == 1 & feat == 1))/sum(sum(feat == 1));
    precision = sum(sum(round(Z) == 1 & feat == 1))/sum(sum(round(Z) == 1));
    f1 = 2*recall*precision/(precision+recall);
end