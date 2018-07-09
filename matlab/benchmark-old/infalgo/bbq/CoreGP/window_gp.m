function GP_y = window_gp(GP_y,params)



% this stuff
[K]=GP_y.covfn(GP_y.hypersample(closestInd).hyperparameters);
Kstst=K(XStar,XStar);
KStarData=K(XStar,XData);
test_sig=wsig;
drop=params.deldrop;
cholK=GP_y.hypersample(closestInd).cholK;
while ~isempty(test_sig) && min(test_sig < params.threshold) && drop < size(XData, 1)
    % The -1 above is because we're just about to add an additional pt
    % on at the next time step
    cholK=downdatechol(cholK,1:params.deldrop);
    KStarData=KStarData(:,1+params.deldrop:end);
    Khalf=linsolve(cholK',KStarData',lowr)';
    test_sig = min(sqrt(diag(Kstst-Khalf*Khalf')));    
    drop=drop+params.deldrop;
end
dropped=drop-params.deldrop;
    