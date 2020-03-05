function [mtv,xx1,xx2] = vbmc_mtv(vp1,vp2,Ns)
%VBMC_MTV Marginal Total Variation distances between two variational posteriors.
%   MTV = VBMC_MTV(VP1,VP2) returns an estimate of the marginal total 
%   variation distances between two variational posterior distributions VP1 
%   and VP2. MTV is a D-element vector whose elements are the total variation
%   distance between the marginal distributions of VP1 and VP2, for each
%   coordinate dimension. 
%
%   The total variation distance between two densities p1 and p2 is:
%       TV(p1, p2) = 1/2 \int | p1(x) - p2(x) | dx
%
%   MTV = VBMC_MTV(VP1,VP2,NS) uses NS random draws to estimate the MTV 
%   (default NS=1e5).
%
%   [MTV,XX1,XX2] = VBMC_MTV(...) returns NS samples from the variational 
%   posteriors VP1 and VP2 as, respectively, NS-by-D matrices XX1 and XX2, 
%   where D is the dimensionality of the problem.
%
%   VP1 and/or VP2 can be N-by-D matrices of samples from variational 
%   posteriors (they do not need have the same number of samples).
%
%   See also VBMC, VBMC_KLDIV, VBMC_PDF, VBMC_RND, VBMC_DIAGNOSTICS.

if nargin < 3 || isempty(Ns); Ns = 1e5; end

% This was removed because the comparison *has* to be in original space,
% given that the transform might change for distinct variational posteriors
% if nargin < 4 || isempty(origflag); origflag = true; end
origflag = true;

if vbmc_isavp(vp1)
    xx1 = vbmc_rnd(vp1,Ns,origflag,1);
    lb1 = vp1.trinfo.lb_orig;
    ub1 = vp1.trinfo.ub_orig;
else
    xx1 = vp1;
    lb1 = -Inf(1,size(vp1,2));
    ub1 = Inf(1,size(vp1,2));
end
if vbmc_isavp(vp2)
    xx2 = vbmc_rnd(vp2,Ns,origflag,1);
    lb2 = vp2.trinfo.lb_orig;
    ub2 = vp2.trinfo.ub_orig;
else
    xx2 = vp2;
    lb2 = -Inf(1,size(vp2,2));
    ub2 = Inf(1,size(vp2,2));
end
    
D = size(xx1,2);
nkde = 2^13;
mtv = zeros(1,D);

% Set bounds for kernel density estimate
lb1_xx = min(xx1); ub1_xx = max(xx1);
range1 = ub1_xx - lb1_xx;
lb1 = max(lb1_xx-range1/10,lb1); 
ub1 = min(ub1_xx+range1/10,ub1);

lb2_xx = min(xx2); ub2_xx = max(xx2);
range2 = ub2_xx - lb2_xx;
lb2 = max(lb2_xx-range2/10,lb2); 
ub2 = min(ub2_xx+range2/10,ub2);

% Compute marginal total variation
for i = 1:D    
    [~,yy1,x1mesh] = kde1d(xx1(:,i),nkde,lb1(i),ub1(i));
    yy1 = yy1/(qtrapz(yy1)*(x1mesh(2)-x1mesh(1))); % Ensure normalization
    
    [~,yy2,x2mesh] = kde1d(xx2(:,i),nkde,lb2(i),ub2(i));
    yy2 = yy2/(qtrapz(yy2)*(x2mesh(2)-x2mesh(1))); % Ensure normalization
        
    f = @(x) abs(interp1(x1mesh,yy1,x,'spline',0) - interp1(x2mesh,yy2,x,'spline',0));
    bb = sort([x1mesh([1,end]),x2mesh([1,end])]);
    for j = 1:3
        xx_range = linspace(bb(j),bb(j+1),1e5);
        mtv(i) = mtv(i) + 0.5*qtrapz(f(xx_range))*(xx_range(2)-xx_range(1));
    end
end

