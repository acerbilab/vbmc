function lcbmax_vec = recompute_lcbmax(gp,optimState,stats,options)
%RECOMPUTE_LCBMAX Recompute moving LCB maximum based on current GP.

N = optimState.Xn;
Xflag = optimState.X_flag;
X = optimState.X(Xflag,:);
y = optimState.y(Xflag);
if isfield(optimState,'S')
    s2 = optimState.S(Xflag).^2;
else
    s2 = [];
end

fmu = NaN(N,1);
fs2 = fmu;
[~,~,fmu(Xflag),fs2(Xflag)] = gplite_pred(gp,X,y,s2);

lcb = fmu - options.ELCBOImproWeight*sqrt(fs2);
lcb_movmax = movmax(lcb,[numel(lcb),0]);

lcbmax_vec = lcb_movmax(stats.N);

end