function y = infbench_goris2015(x,infprob)
%INFBENCH_GORIS2015 Inference benchmark log pdf -- neuronal model from Goris et al. (2015).

if isempty(x)
    if isempty(infprob) % Generate this document        
        fprintf('\n');

        for n = 7:12
            switch n
                case 7;             name = 'm620r12';
                    xmin = [138.8505216444501 2.38317564009549 0.682321320237262 1.1613095607596 1 0.231748337632257 -0.272638945416596 3.10117864852662 72.8822298534178 0.00789002312857097 0.101380347749067 0.693895739234024];
                    fval = -2594.08310420223;
                case 8;             name = 'm620r35';
                    xmin = [227.530092052922 3.00555244729356 2.44308608399358 0.867188243443111 0.886173951591468 -0.039623616648953 -0.463062374186733 1.03689776623743 642.425443794774 0.001 8.65532672683422 0.0851310168322118];
                    fval = -6349.08642324277;
                case 9;             name = 'm624l54';
                    xmin = [219.1171252067330 1.39227769925336 3.03322024598022 0.617344863953975 0.355100712841601 0.067816661803776 -0.12988300841421 1.22284211173093 345.53039808358 0.001 4.29514870404523 0.308744430060413];
                    fval = -6497.67358979187;
                case 10;            name = 'm625r58';    
                    xmin = [305.847145521642 4.7748407736943 1.65588982473794 3.5 0.0559001728606333 0.0448467266543214 -0.192486743740829 4.81143259854959 349.659335938905 0.0700397166151502 0.213318145774832 1.05373239451993];
                    fval = -2118.86479506553;                
                case 11;            name = 'm625r62';    
                    xmin = [124.892643872064 3.69231469595064 2.86789754058724 0.364058707885479 0.00572852306727856 -0.529446086627358 0.410971814338923 4.63088807020964 59810475.5470703 0.001 0.001 0.173543268428183];
                    fval = -301.732736702732;
                case 12;            name = 'm627r58';    
                    xmin = [127.0183192213690 3.48851184430314 3.5 0.897489804982497 0.127319452033804 0.0551773381806372 -0.015460244000334 1.6645293750073 1068.80040238826 0.001 3.84129161213518 0.329767743116606];
                    fval = -5929.98517589428;                
            end                
            
            infprob = infbench_goris2015([],n);
            
            % First, check optimum            
            trinfo = infprob.Data.trinfo;
            opts = optimoptions('fminunc','Display','off');
            
            x0 = xmin(infprob.idxParams);
            x0 = warpvars(x0,'d',trinfo);   % Convert to unconstrained coordinates            
            [xnew,fvalnew] = fminunc(@(x) -infbench_goris2015(x,infprob), x0, opts);
            
            fvalnew = -fvalnew;
            xmin(infprob.idxParams) = warpvars(xnew,'inv',trinfo);
            fval = fvalnew + warpvars(xnew,'logp',trinfo);
                        
            x0 = xmin(infprob.idxParams);
            x0 = warpvars(x0,'d',trinfo);   % Convert to unconstrained coordinates            
            [xnew,fvalnew] = fminunc(@(x) nlogpost(x,infprob), x0, opts);
            
            fvalnew = -fvalnew;
            xmin_post = xmin;
            xmin_post(infprob.idxParams) = warpvars(xnew,'inv',trinfo);
            fval_post = fvalnew + warpvars(xnew,'logp',trinfo);

            fprintf('\t\t\tcase %d\n',n);
            fprintf('\t\t\t\tname = ''%s'';\n\t\t\t\txmin = %s;\n\t\t\t\tfval = %s;\n',name,mat2str(xmin),mat2str(fval));
            fprintf('\t\t\t\txmin_post = %s;\n\t\t\t\tfval_post = %s;\n',mat2str(xmin_post),mat2str(fval_post));
            
        end
           
        
        
    else
        % Initialization call -- define problem and set up data
        n = infprob(1);
        
        % The parameters and their bounds
        % 01 = preferred direction of motion (degrees), unbounded (periodic [0,360]), logical to use most effective stimulus value for family 1, high contrast as starting point
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

        % The most interesting parameters are 01–04, 06, 08, and 12
        
        % Datasets 1-6 are fake neurons
        % (used in parameter recovery analysis of Goris et al. 2015)    
        % Datasets 7-12 are real neurons (cells 9, 16, 45, 64, 65, and 78)    
        switch n
            case {1,2,3,4,5,6}; name = ['fake0', int2str(n)];
                xmin = NaN(1,12);    fval = Inf;
            case 7;             name = 'm620r12';
                xmin = [138.8505216444501 2.38317564009549 0.682321320237262 1.1613095607596 1 0.231748337632257 -0.272638945416596 3.10117864852662 72.8822298534178 0.00789002312857097 0.101380347749067 0.693895739234024];
                fval = -2594.08310420223;
            case 8;             name = 'm620r35';
                xmin = [227.530092052922 3.00555244729356 2.44308608399358 0.867188243443111 0.886173951591468 -0.039623616648953 -0.463062374186733 1.03689776623743 642.425443794774 0.001 8.65532672683422 0.0851310168322118];
                fval = -6349.08642324277;
            case 9;             name = 'm624l54';
                xmin = [219.1171252067330 1.39227769925336 3.03322024598022 0.617344863953975 0.355100712841601 0.067816661803776 -0.12988300841421 1.22284211173093 345.53039808358 0.001 4.29514870404523 0.308744430060413];
                fval = -6497.67358979187;
            case 10;            name = 'm625r58';    
                xmin = [305.847145521642 4.7748407736943 1.65588982473794 3.5 0.0559001728606333 0.0448467266543214 -0.192486743740829 4.81143259854959 349.659335938905 0.0700397166151502 0.213318145774832 1.05373239451993];
                fval = -2118.86479506553;                
            case 11;            name = 'm625r62';    
                xmin = [124.892643872064 3.69231469595064 2.86789754058724 0.364058707885479 0.00572852306727856 -0.529446086627358 0.410971814338923 4.63088807020964 59810475.5470703 0.001 0.001 0.173543268428183];
                fval = -301.732736702732;
            case 12;            name = 'm627r58';    
                xmin = [127.0183192213690 3.48851184430314 3.5 0.897489804982497 0.127319452033804 0.0551773381806372 -0.015460244000334 1.6645293750073 1068.80040238826 0.001 3.84129161213518 0.329767743116606];
                fval = -5929.98517589428;                
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
                
        % nvec = 1:12;
        lb = [0,.05,.1,.1, 0,-1,-1,1,1e-3,1e-3,1e-3,1e-3];  % First variable is periodic
        ub = [360,15,3.5,3.5,1,1,1,6.5,1e9,1e1,1e1,1e1];
        plb =   [90,.5,.3, .3,0,-0.3,  -1,2,10,    1e-3,1e-2,1e-2];
        pub =   [270,10,3.2,  3.2,1,0.3,  1,5,1e5,  1e-1,1,1];
        noise = [];
        
        % LL = -TPGiveBof(xmin, data.sTP, data.sSF);
        % [LL,fval]
        
        idx_params = [1:4,6,8,12];  % Only consider a subset of params
        D = numel(idx_params);
        trinfo = warpvars(D,lb(idx_params),ub(idx_params),plb(idx_params),pub(idx_params));     % Transform to unconstrained space
        y.xBaseFull = xmin;
        xmin = warpvars(xmin(idx_params),'d',trinfo);
        fval = fval + warpvars(xmin,'logp',trinfo);
        
        Mean = zeros(1,D);
        Cov = eye(D);
        Mode = xmin;
                
        y.D = D;
        y.LB = -Inf(1,D);   % Using unconstrained space
        y.UB = Inf(1,D);
        y.PLB = warpvars(plb(idx_params),'d',trinfo);
        y.PUB = warpvars(pub(idx_params),'d',trinfo);
        
        y.lnZ = 0;              % Log normalization factor
        y.Mean = Mean;          % Distribution moments
        y.Cov = Cov;
        y.Mode = Mode;          % Mode of the pdf
        y.ModeFval = fval;
        y.idxParams = idx_params;
        
        priorMean = 0.5*(y.PUB + y.PLB);
        priorSigma2 = (0.5*(y.PUB - y.PLB)).^2;
        priorCov = diag(priorSigma2);
        y.Prior.Mean = priorMean;
        y.Prior.Cov = priorCov;
        
        % Compute each coordinate separately
        y.Post.Mean = zeros(1,D);
        y.Post.Mode = zeros(1,D);
        % range = 5*(y.PUB-y.PLB);
        y.Post.lnZ = NaN;
        y.Post.Cov = eye(D);        
        
        % Save data and coordinate transformation struct
        y.Data = data;
        y.Data.trinfo = trinfo;
        
    end
    
else
    
    % Iteration call -- evaluate objective function
    
    % Transform unconstrained variables to original space
    x_orig = warpvars(x,'i',infprob.Data.trinfo);
    dy = warpvars(x,'logpdf',infprob.Data.trinfo);   % Jacobian correction
    
    xfull = infprob.xBaseFull;
    xfull(infprob.idxParams) = x_orig;
    
    % Compute log likelihood of data (fcn returns nLL)
    LL = -TPGiveBof(xfull, infprob.Data.sTP, infprob.Data.sSF);    
    y = LL - dy;
    
end

end

%--------------------------------------------------------------------------
function y = nlogpost(x,infprob)
    y = -infbench_goris2015(x,infprob);
    infprob.PriorMean = infprob.Prior.Mean;
    infprob.PriorVar = diag(infprob.Prior.Cov)';
    lnp = infbench_lnprior(x,infprob);
    y = y - lnp;
end