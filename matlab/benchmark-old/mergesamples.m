function [X,lls,funccount] = mergesamples(filepattern,MaxIter,Nml)
%MERGESAMPLES Merge MCMC samples from different files.

if nargin < 2 || isempty(MaxIter); MaxIter = Inf; end
if nargin < 3 || isempty(Nml); Nml = 2e4; end

files = dir(filepattern);
M = numel(files);

Xs = [];
lls = [];
funccount = NaN(M,1);
iter = 1;

for iFile = 1:M
    if files(iFile).isdir; continue; end
    filename = files(iFile).name;
    try
        temp = load(filename);
        
        if isempty(Xs) || isempty(lls)
            N = size(temp.Xs,1);
            D = size(temp.Xs,2);
            Xs = NaN(N,D,M);
            lls = NaN(N,M);
        end
        
        Xs(:,:,iter) = temp.Xs;
        lls(:,iter) = temp.lls;        
        funccount(iter) = temp.output.funccount;
        fprintf('%d..', iter);
        iter = iter + 1;
        if iter > MaxIter
            fprintf('\nReached maximum number of files %d.\n', MaxIter);
            break; 
        end
    catch
        fprintf('\nCould not read data from file %s.\n', filename);
    end
end
fprintf('\n');

imax = iter-1;

Xs = Xs(:,:,1:imax);
lls = lls(:,1:imax);
funccount = funccount(1:imax);

% Compute PSRF
[R,Neff] = psrf(Xs);

% Reshape X
[N,D,M] = size(Xs);
X = NaN(N*M,D);
for m = 1:M
    X((1:N)+N*(m-1),:) = Xs(:,:,m);
end

fprintf('R_max = %.3f. Ntot = %d. Neff_min = %.1f. Total funccount = %d.\n',max(R),N*M,min(Neff),sum(funccount));
fprintf('\n\tMean_mcmc = %s;\n\tCov_mcmc = %s;\n', mat2str(mean(X,1)), mat2str(cov(X)));

if Nml > 0
    % Compute approximation of marginal likelihood
    if N > Nml
        idx = round(linspace(1,N,Nml))';
    else
        idx = (1:N)';
    end
    lnZ = vbgmmnormconst('rlr',X(idx,:),lls(idx));
    fprintf('\tlnZ_mcmc = %s;\n', mat2str(lnZ));
end

lls = lls(:);


