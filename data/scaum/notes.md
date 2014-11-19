mat-repmat(mean(mat), size(mat,1), 1) # substract means.
mat./repmat(std(mat), size(mat, 1), 1) # normalize.