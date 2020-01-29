function errorflag = check_quadcoefficients_vbmc(gp)
%CHECK_QUADCOEFFICIENTS_VBMC Check that the quadratic coefficients are negative.

% Extract integrated basis functions coefficients
D = size(gp.X,2);
Nb = numel(gp.post(1).intmean.betabar);
betabar = zeros(Nb,numel(gp.post));
for s = 1:numel(gp.post)        
    betabar(:,s) = gp.post(s).intmean.betabar;
end
% betabar

if gp.intmeanfun == 3
    errorflag = any(betabar(1+D+(1:D),:) >= 0,2)';
elseif gp.intmeanfun == 4
    tril_mat = tril(true(D),-1); 
    tril_vec = tril_mat(:);
    z = zeros(D*D,1); 
    errorflag = false;
    for b = 1:size(betabar,2)
        beta_mat = z;
        beta_mat(tril_vec) = betabar(1+2*D+(1:D*(D-1)/2),b);
        beta_mat = reshape(beta_mat,[D,D]);
        beta_mat = beta_mat + beta_mat' + diag(betabar(1+D+(1:D),b));
        try
            [~,dd] = chol(-beta_mat);
        catch
            dd = 1;
        end
        % dd
        errorflag = errorflag | dd;
    end
end
    
end