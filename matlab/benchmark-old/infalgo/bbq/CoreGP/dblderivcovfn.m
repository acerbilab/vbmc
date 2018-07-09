function K = dblderivcovfn(fnName,tstar,tdata,TimeScale,LengthScale,param)
% Derivative of stationary covariance functions in one dimension with
% respect to 

if nargin==3
    TimeScale=1;
    LengthScale=1;
elseif nargin==4
    LengthScale=1; 
end 

r=abs(tstar-tdata)./TimeScale;
%dr=(2*(tstar>tdata)-1)./TimeScale; % if tstar==tdata, this is actually undefined
dr2=TimeScale.^-2; % dr^2 = constant wrt tstar

switch fnName
    case 'sqdexp'
        % Squared Exponential Covariance Function
        K = LengthScale.^2 .* exp(-1/2*r.^2) .* (r.^2-1) .* dr2;
    case 'ratquad'
        % Rational Quadratic Covariance Function, param is alpha
        if nargin~=6
            param=2; % default value
        end
        K = LengthScale.^2 .* (1 + 1/(2*param)*r.^2).^(-param-2) .* (-1 + (1 + (2*param)^-1) .* r.^2) .* dr2;
    case 'matern'
        % Matern Covariance Function, param is nu
        %(2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*abs(t1-t2)/TimeScale)^nu
        %   * besselk(nu,sqrt(2*nu)*abs(t1-t2)/TimeScale)
        if nargin~=6
            param=3/2; % default value
        end
        if  param==1/2
            % This is the covariance of the Ornstein-Uhlenbeck process
            K = LengthScale.^2 .* exp(-r) .* dr2;          
        elseif param==3/2
            K = LengthScale.^2 .* 3*exp(-sqrt(3)*r) .* (sqrt(3)*r - 1) .* dr2;
        elseif param==5/2
            K = LengthScale.^2 .* -5/3 .* (1 + sqrt(5)*r - 5*r.^2).*exp(-sqrt(5)*r) .* dr2;
        end
end