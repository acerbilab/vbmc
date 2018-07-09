function K = gcov(fnName,hp,varargin)
% Gradient of stationary covariance functions with
% respect to tstar

process_cov_inputs;

if iscell(fnName)
    switch fnName{2}
        case 'periodic'
            rdr=mat2cell2d(0.5*pi*...
                bsxfun(@rdivide, sin(2*arg), InputScale),...
                NData,onesDims)';
        case 'great-circle'
            error('not implemented');
    end
else % assume non-periodic
    rdr=mat2cell2d(...
        bsxfun(@rdivide, insL-insR, InputScale.^2),NData,ones(1,NDims))';
end
%Each element of the cell array dr represents the derivative with respect
%to a different input dimension.






switch hom_fn_Name
    case 'sqdexp'
        % Squared Exponential Covariance Function
        Km = OutputScale.^2 .* -exp(-1/2*r.^2);
    case 'gaussian'
        % Gaussian Covariance Function
        Km = OutputScale.^2 .* (prod(2*pi*InputScale))^(-0.5) * -exp(-1/2*r.^2);
    case 'ratquad'
        % Rational Quadratic Covariance Function, param is alpha
        if isempty(param)
            param=2; % default value
        end
        Km = OutputScale.^2 .* -(1 + 1/(2*param)*r.^2).^(-param-1);
    case 'matern'
        % Matern Covariance Function, param is nu
        %(2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*abs(t1-t2)/InputScale)^nu
        %   * besselk(nu,sqrt(2*nu)*abs(t1-t2)/InputScale)
        if isempty(param)
            param=5/2; % default value
        end
        if  param==1/2
            % This is the covariance of the Ornstein-Uhlenbeck process
            Km = OutputScale.^2 .* -exp(-r) .* r.^(-1);  
            % Gradient undefined at r = 0
        elseif param==3/2
            Km = OutputScale.^2 .* -3 .*exp(-sqrt(3)*r);
        elseif param==5/2
            Km = OutputScale.^2 .* -5/3 .* (1 + sqrt(5)*r).*exp(-sqrt(5)*r);
        end
    case 'poly'
        % Piecewise polynomial covariance functions with compact support
        if isempty(param)
            param=0; % default value
        end
        j=floor(NDims/2)+param+2;
        switch param
            case 0
                Km = -OutputScale.^2 * j*max(1-r,0).^(j-1) .* r.^(-1);
                % Gradient undefined at r = 0
            case 2
                Km = -OutputScale.^2 * max(1-r,0).^(j+1)*(3+j)*(4+j).*(1+r*(1+j))/3;
        end
end

K=cellfun(@(x) reshape(Km.*x,NRows,NCols), rdr,'UniformOutput', false);