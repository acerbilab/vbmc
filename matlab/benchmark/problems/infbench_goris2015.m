function y = infbench_goris2015(x,infprob)
%INFBENCH_GORIS2015 Inference benchmark log pdf -- neuronal model from Goris et al. (2015).

if isempty(x)
    if isempty(infprob) % Generate this document        

    else
        
        % Initialization call -- define problem and set up data

        % The parameters and their bounds
        % 01 = preferred direction of motion (degrees), unbounded (periodic [-180,180]), logical to use most effective stimulus value for family 1, high contrast as starting point
        % 02 = preferred spatial frequency (cycles per degree), values between [.05 15], logical to use most effective stimulus frequency as starting point
        % 03 = aspect ratio 2-D Gaussion, values between [.1 3.5], 1 is reasonable starting point
        % 04 = derivative order in space, values between [.1 3.5], 1 is reasonable starting point
        % 05 = directional selectivity, values between [0 1], 0.5 is reasonable starting point
        % 06 = gain inhibitory channel, values between [-1 1], but majority of cells between [-.2 .2], -0.1 is reasonable starting point
        % 07 = normalization constant, log10 basis, values between [-1 1], 0 is reasonable starting point 
        % 08 = response exponent, values between [1 6.5], 3 is reasonable starting point
        % 09 = response scalar, values between [1e-3 1e9], 4e3 is reasonable starting point (depending on choice of other starting points)
        % 10 = early additive noise, values between [1e-3 1e1], 0.1 is reasonable starting point
        % 11 = late additive noise, values between [1e-3 1e1], 0.1 is reasonable starting point
        % 12 = variance of response gain, values between [1e-3 1e1], 0.1 is reasonable starting point    

        % The most interesting parameters are %01–04, 06, 08, and 12
        
        % Datasets 1-6 are fake neurons
        % (used in parameter recovery analysis of Goris et al. 2015)    
        % Datasets 7-12 are real neurons (cells 9, 16, 45, 64, 65, and 78)    
        switch n
            case {1,2,3,4,5,6}; name = ['fake0', int2str(n)];
            case 7;             name = 'm620r12';
            case 8;             name = 'm620r35';
            case 9;             name = 'm624l54';
            case 10;            name = 'm625r58';    
            case 11;            name = 'm625r62';    
            case 12;            name = 'm627r58';    
        end

        % Add data directory to MATLAB path
        pathstr = fileparts(mfilename('fullpath'));
        addpath([pathstr,filesep(),'goris2015']);
        
        % Load the spatial frequency data
        temp = load([name, '_sf.mat']);
        data.sSF = temp.S;

        % Load the orientation mixture data
        temp = load([name, '_tp.mat']);
        data.sTP = temp.S;    
        
        D = 12;
        % nvec = 1:12;
        lb = {[-180,.05,.1,.1, 0,-1,-1,1,1e-3,1e-3,1e-3,1e-3], 1};  % First variable is periodic
        ub = [180,15,3.5,3.5,1,1,1,6.5,1e9,1e1,1e1,1e1];
        plb =   [-180,.05,.1, .1,0,-0.3,  -1,1,10,    1e-3,1e-2,1e-2];
        pub =   [180,15,3.5,  3.5,1,0.3,  1,6.5,1e5,  1e-1,1,1];
        if isfield(data.sTP,'paramsGen')
            xmin = data.sTP.paramsGen';
        else
            xmin = NaN(1,D);
        end
        noise = [];
        
        trinfo = warpvars(D,lb,ub,plb,pub);        
        
        Mu = zeros(1,D);
        Cov = eye(D);
        Mode = Mu;        
        
        y.D = D;
        y.LB = -Inf(1,D);   % Using unconstrained space
        y.UB = Inf(1,D);
        y.PLB = warpvars(plb,'d',trinfo);
        y.PUB = warpvars(pub,'d',trinfo);
        y.lnZ = 0;        % Log normalization factor
        y.Mu = Mu;
        y.Sigma = Sigma;
        y.Df = Df;
        y.Mean = Mean;        % Distribution moments
        y.Cov = Cov;
        y.Mode = Mode;        % Mode of the pdf
        
        priorMean = 0.5*(y.PUB + y.PLB);
        priorSigma2 = (0.5*(y.PUB - y.PLB)).^2;
        priorCov = diag(priorSigma2);
        y.Prior.Mean = priorMean;
        y.Prior.Cov = priorCov;
        
        % Compute each coordinate separately
        y.Post.Mean = zeros(1,D);
        y.Post.Mode = zeros(1,D);
        % range = 5*(y.PUB-y.PLB);
        y.Post.lnZ = sum(log(Z));
        y.Post.Cov = eye(D);        
        
        % Save data and coordinate transformation struct
        y.Data = data;
        y.Data.trinfo = trinfo;
        
    end
    
else
    
    % Iteration call -- evaluate objective function
    
    % Transform constrained variables to unconstrained space
    x = warpvars(x,'d',probstruct.Data.trinfo);
    dy = warpvars(x,'logpdf',probstruct.Data.trinfo);   % Jacobian correction
    
    y = TPGiveBof(x, probstruct.Data.sTP, probstruct.Data.sSF) + dy;    
    
end


function probstruct = loadprob(probstruct,wrapperfunc,nid)

% Call wrapper function
func = str2func(wrapperfunc);
[nvec,lb,ub,plb,pub,xmin,noise,data] = func(nid);

if iscell(lb)
    probstruct.PeriodicVars = lb{2};
    lb = lb{1};
end
        
% Assign problem-specific information
probstruct.LowerBound = lb(:)';
probstruct.UpperBound = ub(:)';
probstruct.InitRange = [plb(:)'; pub(:)'];
probstruct.TrueMinX = xmin(:)';
probstruct.data = data;

if ~isfinite(noise); noise = []; end
probstruct.NoiseEstimate = noise;
probstruct.func = ['@(x_,probstruct_) ' wrapperfunc '(' num2str(nid) ',x_,probstruct_.data)'];





