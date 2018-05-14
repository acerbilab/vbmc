function read_laplace()
%READ_LAPLACE Read output of Laplace method.

temp = load('laplace@base@1');
Mean_laplace = temp.history{1}.Output.post.Mean;
Cov_laplace = temp.history{1}.Output.post.Cov;
lnZ_laplace = temp.history{1}.Output.post.lnZ;

probstruct = infprob_init(temp.history{1}.ProbSet,temp.history{1}.Prob,temp.history{1}.SubProb,[],1,[]);

Mean_laplace = warpvars(Mean_laplace,'inv',probstruct.trinfo);
Cov_laplace = diag(probstruct.trinfo.delta)*Cov_laplace*diag(probstruct.trinfo.delta);

fprintf('\t\t\tMean_laplace = %s;\n\t\t\tCov_laplace = %s;\n\t\t\tlnZ_laplace = %s;\n', mat2str(Mean_laplace),mat2str(Cov_laplace),mat2str(lnZ_laplace));