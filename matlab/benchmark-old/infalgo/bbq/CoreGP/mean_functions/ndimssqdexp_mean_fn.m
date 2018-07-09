function Mean = ndimssqdexp_mean_fn(hp)

Mu=hp(end-1); % Maybe I should pass in meanPos?
 
Mean=@(Xs) Mu*(sum(abs(Xs(:,end/2+1:end)),2)==0);
% ie. zero for deriv obs, Mu for fn obs