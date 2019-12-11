function gp = gpreupdate(gp,optimState,options)
%GPREUPDATE Quick posterior reupdate of Gaussian process.

[X_train,y_train,s2_train,t_train] = get_traindata_vbmc(optimState,options);
gp.X = X_train;
gp.y = y_train;
gp.s2 = s2_train;   
gp.t = t_train;
gp = gplite_post(gp);            

end