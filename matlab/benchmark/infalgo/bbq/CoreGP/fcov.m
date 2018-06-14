function K = fcov(fnName,hp,varargin)
% K = fcov(fnName,hp,varargin)
% Stationary covariance functions

process_cov_inputs;

switch hom_fn_Name
    case 'sqdexp'
        % Squared Exponential Covariance Function
        K = OutputScale.^2 .* exp(-1/2*r.^2);
    case 'gaussian'
        % Gaussian Covariance Function
        K = OutputScale.^2 .* (prod(2*pi*InputScale^2))^(-0.5) * exp(-1/2*r.^2);
    case 'ratquad'
        % Rational Quadratic Covariance Function, param is alpha
        if isempty(param)
            param=2; % default value
        end
        K = OutputScale.^2 .* (1 + 1/(2*param)*r.^2).^(-param);
    case 'matern'
        % Matern Covariance Function, param is nu
        %(2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*abs(t1-t2)/TimeScale)^nu
        %   * besselk(nu,sqrt(2*nu)*abs(t1-t2)/TimeScale)
        if isempty(param)
            param=5/2; % default value
        end
        if  param==1/2
            % This is the covariance of the Ornstein-Uhlenbeck process
            K = OutputScale.^2 .* exp(-r);            
        elseif param==3/2
            K = OutputScale.^2 .* (1+sqrt(3)*r).*exp(-sqrt(3)*r);
        elseif param==5/2
            K = OutputScale.^2 .* (1 + sqrt(5)*r + 5/3*r.^2).*exp(-sqrt(5)*r);
        end
    case 'poly'
        % Piecewise polynomial covariance functions with compact support
        if isempty(param)
            param=0; % default value
        end
        j=floor(NDims/2)+param+2;
        switch param
            case 0
                K = OutputScale.^2 * max(1-r,0).^j;
            case 1
                K = OutputScale.^2 * max(1-r,0).^(j+1).*((j+1)*r+1);
            case 2
                K = OutputScale.^2 * max(1-r,0).^(j+2).*((j^2+4*j+3)*r.^2+(3*j+6)*r+3)/3;
            case 3
                K = OutputScale.^2 * max(1-r,0).^(j+3).*((j^3+9*j^2+23*j+15)*r.^3+...
                    (6*j^2+36*j+45)*r.^2+(15*j+45)*r+15)/15;
        end
    case 'prodcompact'
        % compact support
        if isempty(param)
            param=1; % default value
        end
        K = (r<1).*(1+r.^param).^(-3).*((1-r).*cos(pi*r)+1/pi*sin(pi*r)); 
    otherwise
        error('Unrecognised covariance type');
end

if matrify_style
    K=reshape(K,NRows,NCols);
end
