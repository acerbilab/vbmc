function ld = logdet(K)
% returns the log-determinant of posdef matrix K

ld = 2*sum(log(diag(chol(K))));