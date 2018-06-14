function K = gTgcov(fnName,hp,varargin)
% Hessian of stationary covariance functions with
% respect to tstar (cols) and input scales (rows)

process_cov_inputs;

if iscell(fnName)
    if strcmp(fnName{2},'periodic')
        
        arg=2*pi*(insL-insR).*InvScales;
        
        r=sqrt(sin(0.5*arg).^2*ones(NDims,1));  
        rdr=mat2cell2d(0.5*pi*InvScales.*sin(arg),...
            NData,onesDims)';
        rdTr=mat2cell2d(-0.25*arg.*sin(arg),...
            NData,ones(1,NDims))';
        
        dmn=kron2d(-pi/2*diag(InputScale.^1),ones(NData,1)).*...
            repmat(sin(arg)+arg.*cos(arg),NDims,1); 
        % the diag here takes care of where we need zeros.
        % Kron is sub-optimal here - does unnecessary
        % multiplication
        dmn=mat2cell(dmn,NDataDims,onesDims);

    end
    fnName=fnName{1};
else % assume non-periodic
    
    InvScales2=InvScales.^2;
    
    r=sqrt((insL-insR).^2*InputScale'.^-2);
    rdr=mat2cell2d((insL-insR).*InvScales2,NData,onesDims)'; % could also use num2cell, but no speed difference
    rdTr=mat2cell2d(-(insL-insR).^2.*InvScales2,NData,ones(1,NDims))';

    dmn=kron2d(-2*diag(InputScale.^-2),ones(NData,1)).*repmat((insL-insR).*InvScales2,NDims,1);
    % the diag here takes care of where we need zeros.
    % Kron is sub-optimal here - does unnecessary
    % multiplication
    dmn=mat2cell(dmn,NDataDims,onesDims);
    
    % The following is too slow!
%     dmn=zeros(NData*NDims,NDims);
%     starts=1:NData*(NDims+1):NData*NDims*(NDims+1);
%     inds=repmat(starts,NData,1)+repmat((0:NData-1)',1,NDims);
%     dmn(inds)=-2*(insL-insR).*InvScales2;
    
    
end



rdr2=mat2cell2d(repmat(cat(1,rdTr{:}),1,NDims).*repmat(cat(2,rdr{:}),NDims,1),NDataDims,onesDims);
%rdr2==(rdr)^2

%Each element of the cell array dr represents the derivative with respect
%to a different input dimension.

switch hom_fn_Name
    case 'sqdexp'
        % Squared Exponential Covariance Function
        const = OutputScale.^2 .* exp(-1/2*r.^2);
        Km = const;
        Kn = -const;
    case 'ratquad'
        % Rational Quadratic Covariance Function, param is alpha
        if isempty(param)
            param=2; % default value
        end
        Km = OutputScale.^2 .* (1 + 1/(2*param)*r.^2).^(-param-2) .* (1+1/param);
        Kn = OutputScale.^2 .* -(1 + 1/(2*param)*r.^2).^(-param-1);
    case 'matern'
        % Matern Covariance Function, param is nu
        %(2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*abs(t1-t2)/InputScale)^nu
        %   * besselk(nu,sqrt(2*nu)*abs(t1-t2)/InputScale)
        if isempty(param)
            param=5/2; % default value
        end
        if  param==1/2
            % This is the covariance of the Ornstein-Uhlenbeck process
            const = OutputScale.^2 .* exp(-r);
            Km = const .* (r.^-2+r.^-3);
            Kn = const .* (-r.^-1); 
            % Hessian undefined at r = 0
        elseif param==3/2
            const = OutputScale.^2 .* -3 * exp(-sqrt(3)*r);
            Km = const .* -sqrt(3) .* r.^-1;
            Kn = const;
            % Hessian undefined at r = 0
        elseif param==5/2
            const = OutputScale.^2 .* 5/3 * exp(-sqrt(5)*r);
            Km = 5*const;
            Kn = -(1+sqrt(5)*r).*const;
        end
    case 'poly'
        % Piecewise polynomial covariance functions with compact support
        if isempty(param)
            param=0; % default value
        end
        j=floor(NDims/2)+param+2;
        switch param
            case 0
                const = OutputScale.^2 * j * max(1-r,0).^(j-2);
                Km = const.*((j-2)*r.^-2+r.^-3);
                Kn = const.*(1-r.^-1);
                % Hessian undefined at r = 0
            case 2
                const = OutputScale.^2 * 1/3 * max(1-r,0).^j * (3+j) * (4+j);
                Km = const.*(1+j)*(2+j);
                Kn = const.*(r-1).*((1+j)*r+1);
        end
end

K=cellfun(@(x,y) reshape(Km.*x+Kn.*y,NRows,NCols),rdr2,dmn,'UniformOutput', false);