function y = quantile1(x,p)
%QUANTILE1 Quantile of a vector.
%   Y = PRCTILE(X,P) returns percentiles of the values in X.  P is a scalar
%   or a vector of percent values.  When X is a vector, Y is the same size
%   as P, and Y(i) contains the P(i)-th percentile.  When X is a matrix,
%   the i-th row of Y contains the P(i)-th percentiles of each column of X.
%   For N-D arrays, PRCTILE operates along the first non-singleton
%   dimension.
%
%   Percentiles are specified using percentages, from 0 to 100.  For an N
%   element vector X, PRCTILE computes percentiles as follows:
%      1) The sorted values in X are taken as the 100*(0.5/N), 100*(1.5/N),
%         ..., 100*((N-0.5)/N) percentiles.
%      2) Linear interpolation is used to compute percentiles for percent
%         values between 100*(0.5/N) and 100*((N-0.5)/N)
%      3) The minimum or maximum values in X are assigned to percentiles
%         for percent values outside that range.
%
%   PRCTILE treats NaNs as missing values, and removes them.
%
%   Examples:
%      y = prctile(x,50); % the median of x
%      y = prctile(x,[2.5 25 50 75 97.5]); % a useful summary of x
%
%   See also IQR, MEDIAN, NANMEDIAN, QUANTILE.

%   Copyright 1993-2016 The MathWorks, Inc.

% If X is empty, return all NaNs.
if isempty(x)
    y = nan(size(p),'like',x);
else
    % Drop X's leading singleton dims, and combine its trailing dims.  This
    % leaves a matrix, and we can work along columns.
    x = x(:);

    x = sort(x,1);
    n = sum(~isnan(x), 1); % Number of non-NaN values
    
    if isequal(p,0.5) % make the median fast
        if rem(n,2) % n is odd
            y = x((n+1)/2,:);
        else        % n is even
            y = (x(n/2,:) + x(n/2+1,:))/2;
        end
    else
        r = p*n;
        k = floor(r+0.5); % K gives the index for the row just before r
        kp1 = k + 1;      % K+1 gives the index for the row just after r
        r = r - k;        % R is the ratio between the K and K+1 rows

        % Find indices that are out of the range 1 to n and cap them
        k(k<1 | isnan(k)) = 1;
        kp1 = bsxfun( @min, kp1, n );

        % Use simple linear interpolation for the valid percentages
        y = (0.5+r).*x(kp1,:)+(0.5-r).*x(k,:);

        % Make sure that values we hit exactly are copied rather than interpolated
        exact = (r==-0.5);
        if any(exact)
            y(exact,:) = x(k(exact),:);
        end

        % Make sure that identical values are copied rather than interpolated
        same = (x(k,:)==x(kp1,:));
        if any(same(:))
            x = x(k,:); % expand x
            y(same) = x(same);
        end

    end

end

end