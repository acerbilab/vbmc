function [m,dm] = gplite_meanfun(hyp,X,meanfun,y,extras)
%GPLITE_MEANFUN Mean function for lite Gaussian Process regression.
%   M = GPLITE_MEANFUN(HYP,X,MEANFUN) computes the GP mean function
%   MEANFUN evaluated at test points X. HYP is a single column vector of mean 
%   function hyperparameters. MEANFUN can be a scalar or a character array
%   specifying the mean function, as follows:
%
%      MEANFUN          MEAN FUNCTION TYPE                  HYPERPARAMETERS
%      0 or 'zero'      zero                                0
%      1 or 'const'     constant                            1
%      2 or 'linear'    linear                              D+1
%      3 or 'quad'      quadratic                           2*D+1
%      4 or 'negquad'   negative quadratic, centered        2*D+1
%      5 or 'posquad'   positive quadratic, centered        2*D+1
%      6 or 'se'        squared exponential                 2*D+2
%      7 or 'negse'     negative squared exponential        2*D+2
%      function_handle  custom                              NMEAN
%
%   MEANFUN can be a function handle to a custom mean function.
%
%   [M,DM] = GPLITE_MEANFUN(HYP,X,MEANFUN) also computes the gradient DM 
%   with respect to GP hyperparamters. DM is a N-by-NMEAN matrix, where
%   each row represent the gradient with respect to the NMEAN hyperparameters
%   for each one of the N test point.
%
%   NMEAN = GPLITE_MEANFUN([],X,MEANFUN) returns the number of mean function
%   hyperparameters requested by mean function MEANFUN.
%
%   [NMEAN,MEANINFO] = GPLITE_MEANFUN([],X,MEANFUN,Y), where X is the matrix
%   of training inputs and Y the matrix of training targets, also returns a 
%   struct MEANINFO with additional information about mean function
%   hyperparameters, with fields: LB (lower bounds); UB (upper bounds); PLB
%   (plausible lower bounds); PUB (plausible upper bounds); x0 (starting
%   point); meanfun (MEANFUN numerical identifier), meanfun_name (MEANFUN
%   name).
%
%   See also GPLITE_COVFUN, GPLITE_NOISEFUN.

if nargin < 4; y = []; end
if nargin < 5; extras = []; end

if isa(meanfun,'function_handle')
    if nargout > 1
        [m,dm] = meanfun(hyp,X,y,extras);
    else
        m = meanfun(hyp,X,y,extras);
    end
    return;
end

if iscell(meanfun); meanfun = meanfun{1}; end

[N,D] = size(X);            % Number of training points and dimension

% Read number of mean function hyperparameters
switch meanfun
    case {0,'0','zero'}
        Nmean = 0;
        meanfun = 0;
    case {1,'1','const'}
        Nmean = 1;
        meanfun = 1;
    case {2,'2','linear'}
        Nmean = 1 + D;
        meanfun = 2;
    case {3,'3','quad'}
        Nmean = 1 + 2*D;        
        meanfun = 3;
    case {4,'4','negquad'}
        Nmean = 1 + 2*D;
        meanfun = 4;
    case {5,'5','posquad'}
        Nmean = 1 + 2*D;
        meanfun = 5;
    case {6,'6','se'}
        Nmean = 2 + 2*D;
        meanfun = 6;
    case {7,'7','negse'}
        Nmean = 2 + 2*D;
        meanfun = 7;
    case {8,'8','negquadse'}
        Nmean = 2 + 4*D;
        meanfun = 8;
    case {9,'9','posquadse'}
        Nmean = 2 + 4*D;
        meanfun = 9;
    case {10,'10','negquadfixiso'}
        Nmean = 2;
        meanfun = 10;
    case {11,'11','posquadfixiso'}
        Nmean = 2;
        meanfun = 11;        
    case {12,'12','negquadfix'}
        Nmean = 1 + D;
        meanfun = 12;
    case {13,'13','posquadfix'}
        Nmean = 1 + D;
        meanfun = 13;
    case {14,'14','negquadsefix'}
        Nmean = 3 + D;
        meanfun = 14;
    case {15,'15','posquadsefix'}
        Nmean = 3 + D;
        meanfun = 15;
    case {16,'16','negquadonly'}
        Nmean = D;
        meanfun = 16;
    case {17,'17','posquadonly'}
        Nmean = D;
        meanfun = 17;        
    case {18,'18','negquadfixonly'}
        Nmean = D;
        meanfun = 18;
    case {19,'19','posquadfixonly'}
        Nmean = D;
        meanfun = 19;
    case {20,'20','negquadlinonly'}
        Nmean = 2*D;
        meanfun = 20;
    case {21,'21','posquadlinonly'}
        Nmean = 2*D;
        meanfun = 21;
    otherwise
        if isnumeric(meanfun); meanfun = num2str(meanfun); end
        error('gplite_meanfun:UnknownMeanFun',...
            ['Unknown mean function identifier: [' meanfun '].']);
end

% Return number of mean function hyperparameters and additional info
if ischar(hyp)
    m = Nmean;
    if nargout > 1
        ToL = 1e-6;
        Big = exp(3);
        w = max(X) - min(X);                    % Width
        
        dm.LB = -Inf(1,Nmean);
        dm.UB = Inf(1,Nmean);
        dm.PLB = -Inf(1,Nmean);
        dm.PUB = Inf(1,Nmean);
        dm.x0 = NaN(1,Nmean);
        dm.extras = [];

        
        if numel(y) <= 1; y = [0;1]; end
        
        if meanfun >= 1 && meanfun <= 15                     % m0
            h = max(y) - min(y);    % Height
            dm.LB(1) = min(y) - 0.5*h;
            dm.UB(1) = max(y) + 0.5*h;
            dm.PLB(1) = quantile1(y,0.1);
            dm.PUB(1) = quantile1(y,0.9);
            dm.x0(1) = median(y);
        end
        
        if meanfun >= 2 && meanfun <= 3     % a for linear part
            delta = w./h;
            dm.LB(2:D+1) = -delta*Big;
            dm.UB(2:D+1) = delta*Big;
            dm.PLB(2:D+1) = -delta;
            dm.PUB(2:D+1) = delta;
            if meanfun == 3
                dm.LB(D+2:2*D+1) = -(delta*Big).^2;
                dm.UB(D+2:2*D+1) = (delta*Big).^2;                
                dm.PLB(D+2:2*D+1) = -delta.^2;
                dm.PUB(D+2:2*D+1) = delta.^2;                
            end
            
        elseif meanfun >= 4 && meanfun <= 15
            
            % Redefine limits for m0 (meaning depends on mean func type)
            h = max(y) - min(y);    % Height
            switch meanfun
                case {4,10}
                    dm.LB(1) = min(y);
                    dm.UB(1) = max(y) + h;
                    dm.PLB(1) = median(y);
                    dm.PUB(1) = max(y);
                    dm.x0(1) = quantile1(y,0.9);
                case {5,11}
                    dm.LB(1) = min(y) - h;
                    dm.UB(1) = max(y);
                    dm.PLB(1) = min(y);
                    dm.PUB(1) = median(y);
                    dm.x0(1) = quantile1(y,0.1);
                case 6
                    dm.LB(1) = min(y) - h;
                    dm.UB(1) = max(y);
                    dm.PLB(1) = min(y);
                    dm.PUB(1) = median(y);
                    dm.x0(1) = quantile1(y,0.1);
                case 7
                    dm.LB(1) = min(y);
                    dm.UB(1) = max(y) + h;
                    dm.PLB(1) = median(y);
                    dm.PUB(1) = max(y);
                    dm.x0(1) = quantile1(y,0.9);                    
                case {8,9}
                    dm.LB(1) = min(y) - h;
                    dm.UB(1) = max(y) + h;
                    dm.PLB(1) = min(y);
                    dm.PUB(1) = max(y);
                    dm.x0(1) = median(y);
                case {14,15}
                    dm.LB(1) = min(y) - h;
                    dm.UB(1) = max(y) + h;
                    dm.PLB(1) = min(y);
                    dm.PUB(1) = max(y);
                    dm.x0(1) = median(y);
            end
            
            if meanfun >= 4 && meanfun <= 9
                dm.LB(2:D+1) = min(X) - 0.5*w;          % xm
                dm.UB(2:D+1) = max(X) + 0.5*w;
                dm.PLB(2:D+1) = min(X);
                dm.PUB(2:D+1) = max(X);
                dm.x0(2:D+1) = median(X);

                dm.LB(D+2:2*D+1) = log(w) + log(ToL);   % omega
                dm.UB(D+2:2*D+1) = log(w) + log(Big);
                dm.PLB(D+2:2*D+1) = log(w) + 0.5*log(ToL);
                dm.PUB(D+2:2*D+1) = log(w);
                dm.x0(D+2:2*D+1) = log(std(X));
            end
            
            if meanfun == 6 || meanfun == 7                
                dm.LB(2*D+2) = log(h) + log(ToL);   % h
                dm.UB(2*D+2) = log(h) + log(Big);
                dm.PLB(2*D+2) = log(h) + 0.5*log(ToL);
                dm.PUB(2*D+2) = log(h);
                dm.x0(2*D+2) = log(std(y));   
            end
            
            if meanfun == 8 || meanfun == 9                
                dm.LB(2*D+1+(1:D)) = min(X) - 0.5*w;          % xm_se
                dm.UB(2*D+1+(1:D)) = max(X) + 0.5*w;
                dm.PLB(2*D+1+(1:D)) = min(X);
                dm.PUB(2*D+1+(1:D)) = max(X);
                [~,idx_max] = max(y);
                dm.x0(2*D+1+(1:D)) = X(idx_max,:);

                dm.LB(3*D+1+(1:D)) = log(w) + log(ToL);   % omega_se
                dm.UB(3*D+1+(1:D)) = log(w) + log(Big);
                dm.PLB(3*D+1+(1:D)) = log(w) + 0.5*log(ToL);
                dm.PUB(3*D+1+(1:D)) = log(w);
                dm.x0(3*D+1+(1:D)) = log(std(X));
                
                dm.LB(4*D+2) = -Big*h;   % h_se
                dm.UB(4*D+2) = Big*h;
                dm.PLB(4*D+2) = -h;
                dm.PUB(4*D+2) = h;
                dm.x0(4*D+2) = min(std(y),h);   
            end
            
            if meanfun == 10 || meanfun == 11
                dm.LB(2) = min(log(w)) + log(ToL);  % omega isotropic
                dm.UB(2) = max(log(w)) + log(Big);
                dm.PLB(2) = min(log(w)) + 0.5*log(ToL);
                dm.PUB(2) = max(log(w));
                dm.x0(2) = mean(log(std(X)));                
            end
            
            if meanfun == 12 || meanfun == 13 || meanfun == 14 || meanfun == 15
                dm.LB(2:D+1) = log(w) + log(ToL);   % omega
                dm.UB(2:D+1) = log(w) + log(Big);
                dm.PLB(2:D+1) = log(w) + 0.5*log(ToL);
                dm.PUB(2:D+1) = log(w);
                dm.x0(2:D+1) = log(std(X));
            end
            
            if meanfun == 14 || meanfun == 15
                dm.LB(D+2) = log(0.01);      % alpha_se
                dm.UB(D+2) = log(10);
                dm.PLB(D+2) = log(0.1);
                dm.PUB(D+2) = log(1);
                dm.x0(D+2) = log(0.5);
                
                dm.LB(D+3) = log(1e-3);      % h_se
                dm.UB(D+3) = log(1e4);
                dm.PLB(D+3) = log(0.1);
                dm.PUB(D+3) = log(100);
                dm.x0(D+3) = log(1);
            end
            
        elseif meanfun >=16 && meanfun <= 19
            dm.LB(1:D) = log(w) + log(ToL);   % omega
            dm.UB(1:D) = log(w) + log(Big);
            dm.PLB(1:D) = log(w) + 0.5*log(ToL);
            dm.PUB(1:D) = log(w);
            dm.x0(1:D) = log(std(X));
            
        elseif meanfun >= 20 && meanfun <= 21
            dm.LB(1:D) = min(X) - 0.5*w;          % xm
            dm.UB(1:D) = max(X) + 0.5*w;
            dm.PLB(1:D) = min(X);
            dm.PUB(1:D) = max(X);
            dm.x0(1:D) = median(X);                
                
            dm.LB(D+(1:D)) = log(w) + log(ToL);   % omega
            dm.UB(D+(1:D)) = log(w) + log(Big);
            dm.PLB(D+(1:D)) = log(w) + 0.5*log(ToL);
            dm.PUB(D+(1:D)) = log(w);
            dm.x0(D+(1:D)) = log(std(X));            
        end
        
        % Fixed quadratic mean location
        if meanfun >= 10 && meanfun <= 15 || meanfun >= 18 && meanfun <= 19
            if any(meanfun == [10,12,14,18])   % find max/min
                [~,idx] = max(y);
            else
                [~,idx] = min(y);
            end                
            dm.extras = X(idx,:);
        end
                
        % Plausible starting point
        idx_nan = isnan(dm.x0);
        dm.x0(idx_nan) = 0.5*(dm.PLB(idx_nan) + dm.PUB(idx_nan));
        
        dm.meanfun = meanfun;
        switch meanfun
            case 0; dm.meanfun_name = 'zero';
            case 1; dm.meanfun_name = 'const';
            case 2; dm.meanfun_name = 'linear';
            case 3; dm.meanfun_name = 'quad';
            case 4; dm.meanfun_name = 'negquad';
            case 5; dm.meanfun_name = 'posquad';
            case 6; dm.meanfun_name = 'se';
            case 7; dm.meanfun_name = 'negse';
            case 8; dm.meanfun_name = 'negquadse';
            case 9; dm.meanfun_name = 'posquadse';
            case 10; dm.meanfun_name = 'negquadfixiso';
            case 11; dm.meanfun_name = 'posquadfixiso';
            case 12; dm.meanfun_name = 'negquadfix';
            case 13; dm.meanfun_name = 'posquadfix';
            case 14; dm.meanfun_name = 'negquadsefix';
            case 15; dm.meanfun_name = 'posquadsefix';
            case 16; dm.meanfun_name = 'negquadonly';
            case 17; dm.meanfun_name = 'posquadonly';
            case 18; dm.meanfun_name = 'negquadfixonly';
            case 19; dm.meanfun_name = 'posquadfixonly';
            case 20; dm.meanfun_name = 'negquadlinonly';
            case 21; dm.meanfun_name = 'posquadlinonly';
        end
        
    end
    
    return;
end

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

if Nhyp ~= Nmean
    error('gplite_meanfun:WrongMeanHyp', ...
        ['Expected ' num2str(Nmean) ' mean function hyperparameters, ' num2str(Nhyp) ' passed instead.']);
end
if Ns > 1
    error('gplite_meanfun:nosampling', ...
        'Mean function output is available only for one-sample hyperparameter inputs.');
end

% Compute mean function gradient wrt hyperparameters only if requested
compute_grad = nargout > 1;

if compute_grad     % Allocate space for gradient
    dm = zeros(N,Nmean);    
end

% Compute mean function    
switch meanfun
    case 0  % Zero
        m = zeros(N,1);
        if compute_grad; dm = []; end
    case 1  % Constant
        m0 = hyp(1);
        m = m0*ones(N,1);
        if compute_grad; dm = ones(N,1); end
    case 2  % Linear
        m0 = hyp(1);
        a = hyp(1+(1:D))';
        m = m0 + sum(bsxfun(@times,a,X),2);
        if compute_grad
            dm(:,1) = ones(N,1); 
            dm(:,2:D+1) = X; 
        end
    case 3  % Quadratic
        m0 = hyp(1);
        a = hyp(1+(1:D))';
        b = hyp(1+D+(1:D))';
        m = m0 + sum(bsxfun(@times,a,X),2) + sum(bsxfun(@times,b,X.^2),2);
        if compute_grad
            dm(:,1) = ones(N,1); 
            dm(:,2:D+1) = X; 
            dm(:,D+2:2*D+1) = X.^2; 
        end
    case {4,5}  % Negative (4) and positive (5) quadratic
        if meanfun == 4; sgn = -1; else; sgn = 1; end
        m0 = hyp(1);
        xm = hyp(1+(1:D))';
        omega = exp(hyp(D+1+(1:D)))';
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        m = m0 + (sgn*0.5)*sum(z2,2);
        if compute_grad
            dm(:,1) = ones(N,1);
            dm(:,2:D+1) = (-sgn)*bsxfun(@rdivide,bsxfun(@minus,X,xm), omega.^2);
            dm(:,D+2:2*D+1) = (-sgn)*z2;        
        end
    case {6,7}  % Squared exponential (6) and negative squared exponential (7) 
        m0 = hyp(1);
        xm = hyp(1+(1:D))';
        omega = exp(hyp(D+1+(1:D)))';
        h = exp(hyp(2*D+2));
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        if meanfun == 6
            se = h*exp(-0.5*sum(z2,2));
        else
            se = -h*exp(-0.5*sum(z2,2));
        end
        m = m0 + se;
        if compute_grad
            dm(:,1) = ones(N,1);
            dm(:,2:D+1) = bsxfun(@times, bsxfun(@rdivide,bsxfun(@minus,X,xm), omega.^2), se);
            dm(:,D+2:2*D+1) = bsxfun(@times, z2, se);
            dm(:,2*D+2) = se;
        end
    case {8,9}  % Sum of negative (8) or positive (9) quadratic and squared exponential
        if meanfun == 8; sgn = -1; else; sgn = 1; end        
        m0 = hyp(1);
        xm = hyp(1+(1:D))';
        omega = exp(hyp(D+1+(1:D)))';
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        
        xm_se = hyp(2*D+1+(1:D))';
        omega_se = exp(hyp(3*D+1+(1:D)))';
        h_se = hyp(4*D+2);
        z2_se = bsxfun(@rdivide,bsxfun(@minus,X,xm_se),omega_se).^2;
        se0 = exp(-0.5*sum(z2_se,2));
        se = h_se*se0;
        
        m = m0 + (sgn*0.5)*sum(z2,2) + se;
        if compute_grad
            dm(:,1) = ones(N,1);
            dm(:,2:D+1) = (-sgn)*bsxfun(@rdivide,bsxfun(@minus,X,xm), omega.^2);
            dm(:,D+2:2*D+1) = (-sgn)*z2;        
            dm(:,2*D+1+(1:D)) = bsxfun(@times, bsxfun(@rdivide,bsxfun(@minus,X,xm_se), omega_se.^2), se);
            dm(:,3*D+1+(1:D)) = bsxfun(@times, z2_se, se);
            dm(:,4*D+2) = se0;
        end
    case {10,11}    % Negative (10) and positive (11) quadratic, fixed-location and isotropic
        if meanfun == 10; sgn = -1; else; sgn = 1; end
        m0 = hyp(1);
        xm = extras(1:D);
        omega = exp(hyp(2));
        z2 = bsxfun(@minus,X/omega,xm/omega).^2;
        m = m0 + (sgn*0.5)*sum(z2,2);
        if compute_grad
            dm(:,1) = ones(N,1);
            dm(:,2) = (-sgn)*sum(z2,2);        
        end
    case {12,13}  % Negative (12) and positive (13) quadratic, fixed-location
        if meanfun == 12; sgn = -1; else; sgn = 1; end
        m0 = hyp(1);
        xm = extras(1:D);
        omega = exp(hyp(2:D+1))';
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        m = m0 + (sgn*0.5)*sum(z2,2);
        if compute_grad
            dm(:,1) = ones(N,1);
            dm(:,2:D+1) = (-sgn)*z2;        
        end
    case {14,15}  % Sum of negative (14) or positive (15) quadratic and squared exponential, constrained
        if meanfun == 14; sgn = -1; else; sgn = 1; end        
        m0 = hyp(1);
        xm = extras(1:D);
        omega = exp(hyp(1+(1:D)))';
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        
        alpha_se = exp(hyp(D+2));   % Rescaling for the squared exponential
        % omega_se = alpha_se*omega;
        h_se = exp(hyp(D+3));
        z2_se = z2 ./ alpha_se^2;
        se0 = exp(-0.5*sum(z2_se,2));
        se = h_se*se0;
        
        m = (m0 + sgn*h_se) + (sgn*0.5)*sum(z2,2) - sgn*se;
        if compute_grad
            dm(:,1) = ones(N,1);
            dm(:,2:D+1) = (-sgn)*z2 - sgn*bsxfun(@times, z2_se, se);      
            dm(:,D+2) = -sgn*sum(z2_se,2).*se;
            dm(:,D+3) = sgn*h_se - sgn*h_se*se0;
        end        
    case {16,17}  % Negative (16) and positive (17) quadratic only
        if meanfun == 16; sgn = -1; else; sgn = 1; end
        omega = exp(hyp(1:D))';
        z2 = bsxfun(@rdivide,X,omega).^2;
        m = sgn*0.5*sum(z2,2);
        if compute_grad
            dm(:,1:D) = (-sgn)*z2;        
        end
    case {18,19}  % Negative (18) and positive (19) quadratic, fixed-location, no shift
        if meanfun == 18; sgn = -1; else; sgn = 1; end
        xm = extras(1:D);
        omega = exp(hyp(1:D))';
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        m = (sgn*0.5)*sum(z2,2);
        if compute_grad
            dm(:,1:D) = (-sgn)*z2;        
        end
    case {20,21}  % Negative (20) and positive (21) quadratic, no shift
        if meanfun == 20; sgn = -1; else; sgn = 1; end
        xm = hyp((1:D))';
        omega = exp(hyp(D+(1:D)))';
        z2 = bsxfun(@rdivide,bsxfun(@minus,X,xm),omega).^2;
        m = (sgn*0.5)*sum(z2,2);
        if compute_grad
            dm(:,1:D) = (-sgn)*bsxfun(@rdivide,bsxfun(@minus,X,xm), omega.^2);
            dm(:,D+(1:D)) = (-sgn)*z2;        
        end
                
        
end

end