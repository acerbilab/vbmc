function read_laplace()
%READ_LAPLACE Read output of Laplace method.

temp = load('laplace@base@1');
Mean_laplace = temp.history{1}.Output.post.Mean;
Cov_laplace = temp.history{1}.Output.post.Cov;
lnZ_laplace = temp.history{1}.Output.post.lnZ;
fprintf('\t\t\tMean_laplace = %s;\n\t\t\tCov_laplace = %s;\n\t\t\tlnZ_laplace = %s;\n', mat2str(Mean_laplace),mat2str(Cov_laplace),mat2str(lnZ_laplace));