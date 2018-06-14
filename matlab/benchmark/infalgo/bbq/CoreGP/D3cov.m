function K = D3cov(fnName,hp,varargin)
% (Cell) Array of third derivatives of stationary covariance functions with
% respect to tstar

NDims=length(varargin)/2;
if NDims~=floor(NDims)
    error('Inputs must have identical dimension');
end
NRows=size(varargin{1},1);
NCols=size(varargin{1},2);
NData=NRows*NCols;

onesDims=ones(1,NDims);
NDataDims(1,NDims)=NData;
NDataDims(:)=NDataDims(end);

InputScale=onesDims;
OutputScale=1;
param=[];
if ~isempty(hp)
    InputScale=hp{1};
    if length(hp)>1
        OutputScale=hp{2};
        if length(hp)>2
            param=hp{3};
        end
    end
end

% ins={[1111,1112;1121,1122],[1211,1212;1221,1222],[1311,1312;1321,1322],...
%    [9111,9112;9121,9122],[9211,9212;9221,9222],[9311,9312;9321,9322]};

ins=reshape(varargin,NDims,2); % break into L and R columns
insL=[ins{:,1}];
insL=reshape(insL,NRows*NCols,NDims); % put rows and cols into a single vector
insR=[ins{:,2}];
insR=reshape(insR,NRows*NCols,NDims);

InvScales=repmat(InputScale.^-1,NData,1);

if iscell(fnName)
    if strcmp(fnName{2},'periodic')
        
        arg=2*pi*(insL-insR).*InvScales;
        
        r=sqrt(sin(0.5*arg).^2*ones(NDims,1));  
        rdr=0.5*pi*InvScales.*sin(arg);     
        
        dmn=kron2d(diag(pi^2*InputScale.^-2),ones(NData,1)).*...
            repmat(cos(arg),NDims,1);
        % Kron is sub-optimal here - does unnecessary
        % multiplication

        ddd=zeros(NData*NDims,NDims,NDims);
        starts=1:NData*(NDims^2+NDims+1):NData*NDims*(NDims^2+NDims+1);
        inds=repmat(starts,NData,1)+repmat((0:NData-1)',1,NDims);
        ddd(inds)=-2*pi^3*InvScales.^3.*sin(arg);
        % This should be like a 3-d block diagonal matrix;
        ddd=mat2cell3d(ddd,NDataDims,onesDims,onesDims);
        
    end
    fnName=fnName{1};
else % assume non-periodic
    
    InvScales2=InvScales.^2;

    r=sqrt((insL-insR).^2*InputScale'.^-2);
    rdr=(insL-insR).*InvScales2; % could also use num2cell, but no speed difference

    dmn=kron2d(diag(InputScale.^-2),ones(NData,1));
    % Kron is sub-optimal here - does unnecessary multiplication
    
%     dmn=zeros(NData*NDims,NDims);
%     starts=1:NData*(NDims+1):NData*NDims*(NDims+1);
%     inds=repmat(starts,NData,1)+repmat((0:NData-1)',1,NDims);
%     dmn(inds)=InvScales2;
    
    ddd=0;
end

rdrcol=reshape(rdr,NData*NDims,1);
rdrtower=reshape(rdr,NData,1,NDims);

rdr3=mat2cell3d(...
      repmat(rdrcol,1,NDims,NDims)...
    .*repmat(rdr,NDims,1,NDims)...
    .*repmat(rdrtower,NDims,NDims,1)...
    ,NDataDims,onesDims,onesDims);
%rdr3==(rdr)^3

rdrdmn1=repmat(dmn,1,1,NDims).*repmat(rdrtower,NDims,NDims,1);
rdrdmn2=repmat(reshape(dmn,NData*NDims,1,NDims),1,NDims,1).*repmat(rdr,NDims,1,NDims);
rdrdmn3=repmat(reshape(dmn,NData,NDims,NDims),NDims,1,1).*repmat(rdrcol,1,NDims,NDims);

rdrdd=mat2cell3d(rdrdmn1+rdrdmn2+rdrdmn3,NDataDims,onesDims,onesDims);

%Each element of the cell array dr represents the derivative with respect
%to a different input dimension.

switch fnName
    case 'sqdexp'
        % Squared Exponential Covariance Function
        const = OutputScale.^2 .* exp(-1/2*r.^2);
        Krdr3 = -const;
        Krdrdd = const;
    case 'ratquad'
        % Rational Quadratic Covariance Function, param is alpha
        if isempty(param)
            param=2; % default value
        end

    case 'matern'
        % Matern Covariance Function, param is nu
        %(2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*abs(t1-t2)/InputScale)^nu
        %   * besselk(nu,sqrt(2*nu)*abs(t1-t2)/InputScale)
        if isempty(param)
            param=3/2; % default value
        end
        if  param==1/2
            % This is the covariance of the Ornstein-Uhlenbeck process
            const = OutputScale.^2 .* exp(-r);
            Krdr3 = -const.*(r.^-3+3*r.^-4+3*r.^-5);
            Krdrdd = const.*(r.^-2+r.^-3);
            % Hessian undefined at r = 0
        elseif param==3/2
            const = OutputScale.^2 .* -3 * exp(-sqrt(3)*r);
            Krdr3 = const.*(3*r.^-2+sqrt(3)*r.^-3);
            Krdrdd = -const.*sqrt(3).*r.^-1;
            % Hessian undefined at r = 0
        elseif param==5/2
            const = OutputScale.^2 .* 5/3 * exp(-sqrt(5)*r);
            Krdr3 = -const.*5*sqrt(5)*r.^-1;
            Krdrdd = 5*const;
        end
end
if isscalar(ddd) && ddd==0
    K=cellfun(@(x,y) reshape(Krdr3.*x+Krdrdd.*y,NRows,NCols),rdr3,rdrdd,'UniformOutput', false);
else

    switch fnName
        case 'sqdexp'
            % Squared Exponential Covariance Function
            Kddd = -const;
        case 'ratquad'
            % Rational Quadratic Covariance Function, param is alpha
            if isempty(param)
                param=2; % default value
            end

        case 'matern'
            % Matern Covariance Function, param is nu
            %(2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*abs(t1-t2)/InputScale)^nu
            %   * besselk(nu,sqrt(2*nu)*abs(t1-t2)/InputScale)
            if isempty(param)
                param=3/2; % default value
            end
            if  param==1/2
                % This is the covariance of the Ornstein-Uhlenbeck process
                Kddd = -const.*r.^-1;
                % Hessian undefined at r = 0
            elseif param==3/2
                Kddd = const;
                % Hessian undefined at r = 0
            elseif param==5/2
                Kddd = -const.*(1+sqrt(5)*r);
            end
    end
    
    K=cellfun(@(x,y,z) reshape(Krdr3.*x+Krdrdd.*y+Kddd.*z,NRows,NCols),rdr3,rdrdd,ddd,'UniformOutput', false);
end