function tx = log_transform(x, gamma)

tx = log(bsxfun(@rdivide, x, gamma) + 1);


