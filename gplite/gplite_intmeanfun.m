function H = gplite_intmeanfun(X,intmeanfun,y,extras)
%GPLITE_INTMEANFUN Integrated mean function for lite Gaussian Process regression.
%   M = GPLITE_INTMEANFUN(HYP,X,MEANFUN) computes the GP mean function
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

[N,D] = size(X);            % Number of training points and dimension

switch intmeanfun
    case {1,'1','const'}
        intmeanfun = 1;
        Nb = 1;
    case {2,'2','linear'}
        intmeanfun = 2;
        Nb = 1 + D;
    case {3,'3','quadratic'}
        intmeanfun = 3;
        Nb = 1 + 2*D;
    case {4,'4','full','fullquad','fullquadratic'}
        intmeanfun = 4;
        Nb = 1 + 2*D + D*(D-1)/2;
    otherwise
        if isnumeric(intmeanfun); intmeanfun = num2str(intmeanfun); end
        error('gplite_intmeanfun:UnknownMeanFun',...
            ['Unknown integrated mean function identifier: [' intmeanfun '].']);
end

H = zeros(Nb,N);

if intmeanfun >= 1
    H(1,:) = 1;
end
if intmeanfun >= 2
    H(2:D+1,:) = X';
end
if intmeanfun >= 3
    H(D+2:2*D+1,:) = X'.^2;
end
if intmeanfun >= 4
    idx = 0;
    for d = 1:D-1
        H(1+2*D+idx+(1:D-d),:) = bsxfun(@times,X(:,d)',X(:,d+1:D)');
        idx = idx + D-d;
    end
end
    
end


        