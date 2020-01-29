function gp = gpreupdate(gp,optimState,options)
%GPREUPDATE Quick posterior reupdate of Gaussian process.

[X_train,y_train,s2_train,t_train] = get_traindata_vbmc(optimState,options);
gp.X = X_train;
gp.y = y_train;
gp.s2 = s2_train;   
gp.t = t_train;
gp = gplite_post(gp);            

if gp.intmeanfun == 3 || gp.intmeanfun == 4
    errorflag = check_quadcoefficients_vbmc(gp);
    if errorflag
        gp.meanfun = optimState.gpMeanfun;
        gp.intmeanfun = [];
        
        for s = 1:numel(gp.post)
            betabar = gp.post(s).intmean.betabar;
            hyp = gp.post(s).hyp;
            
            switch gp.meanfun
                case 4
                    omega2 = -1./betabar(1+D+(1:D));
                    xm = omega2.*betabar(1+(1:D));
                    m0 = betabar(1) + 0.5*xm.^2./omega2;
                    hyp_mean = [m0; xm(:); 0.5*log(omega2(:))];                    
                    hypnew = [hyp(1:gp.Ncov+gp.Nnoise); hyp_mean(:); hyp(gp.Ncov+gp.Nnoise+1:end)];
            end            
            gp.post(s).hyp = hypnew;            
        end
        
        % Recompute GP without integrated mean function
        gp = gplite_post(gp);            
    end
end
    
end