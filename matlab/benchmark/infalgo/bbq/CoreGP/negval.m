function [f,out2,out3,out4,out5,out6,out7,out8,out9] = ...
    negval(XStar,gp,sample,min_so_far,outputGP)
% Compute the expected value of taking an observation of a function at
% XStar, given that we have already observed YData at XData. gp
% describes our covariance structure for the GP we possess over the
% function.

% the mean m and covariance C (and their gradients & Hessians) of our GP
% evaluated at XStar

if nargin<3
    sample=1;
end
if nargin<4 || isempty(min_so_far)
    min_so_far=min(gp.y_data);
end
if nargin<5
    outputGP=0;
    % The function will not output the computed mean and covariances of the
    % GP
end

if (~outputGP && nargout == 1) || (outputGP && nargout <= 3)
    [m,D]=posterior_gp(XStar,gp,sample,...
        {'jitter_corrected','var_not_cov'});
elseif (~outputGP && nargout == 2) || (outputGP && nargout <= 6)
    [m,C,gm,gC]=posterior_gp(XStar,gp,sample,...
        {'jitter_corrected','var_not_cov'});
    
    gm=cell2mat2d(gm);
    D=diag(C);
    gD=cellfun(@diag,gC);
elseif (~outputGP && nargout == 3) || (outputGP && nargout <= 9)
    [m,C,gm,gC,Hm,HC]=posterior_gp(XStar,gp,sample,...
        {'jitter_corrected','var_not_cov'}); 
    D=diag(C);
    gD=cellfun(@diag,gC,'UniformOutput',0);
    HD=cellfun(@diag,HC,'UniformOutput',0);
end

%finds the diagonal for each derivative

%D=max(D,eps^2); % effectively jitter

argterm=(min_so_far-m)./(sqrt(2*D)); % scalar
f = 0.5*(min_so_far + m) ...
    + 0.5*(m-min_so_far).*erf(argterm) ...
    - sqrt(D/(2*pi)).*exp(-argterm.^2);

% Compute the scalar value at XStar

if outputGP
    out2=m;
    out3=D;
end

if (~outputGP && nargout > 1) || (outputGP && nargout > 3)  
    
    % fun called with two output arguments
    
    NDims=length(gm);
    
    g = 0.5*gm*(1+erf(argterm))-0.5*gD*(2*pi*D)^(-0.5)*exp(-argterm.^2);
    
%     g = 0.5*(2*pi*repmat(D,NDims,1)).^(-0.5).*gD...
%         .*repmat(exp(-argterm.^2),NDims,1)...
%         +0.5*repmat(erfc(argterm),NDims,1).*gm; 
%     g=-g;
    % Gradient vector of the value evaluated at XStar

    out2=g;
    if outputGP
        out3=m;
        out4=D;
        out5=gm;
        out6=gD;
    end
   
    if (~outputGP && nargout > 2) || (outputGP && nargout > 6) % fun called with three output arguments
                
        % I have not-rechecked the below is correct, these are the old eqns
        % from negvalue.
        m = -m;
        mx = -min_so_far;
        argterm=(mx-m)./(sqrt(2*D)); % scalar
        
        term1L=repmat(gD,1,NDims);
        term1R=repmat(gD',NDims,1);
        term2L=repmat((gm+...
                    0.5*gD.*repmat(D.^(-1).*(mx-m),NDims,1)),1,NDims);
        term2R=repmat((gm'+...
                    0.5*gD'.*repmat(D.^(-1).*(mx-m),1,NDims)),NDims,1);
        
        Hm=cell2mat2d(Hm);
        HD=cell2mat2d(HD);

        H = 0.5*Hm.*repmat(erfc(argterm),NDims,NDims)+...
            repmat((2*pi*D.^3).^(-0.5).*exp(-argterm.^2),NDims,NDims).*(...
            -0.25*term1L.*term1R+repmat(D,NDims,NDims).*(0.5*HD+...
            term2L.*term2R));
        H=-H;
        % Hessian matrix evaluated at XStar

        out3=H;
        if outputGP
            out4=m;
            out5=D;
            out6=gm;
            out7=gD;
            out8=Hm;
            out9=HD;
        end
    end
end



