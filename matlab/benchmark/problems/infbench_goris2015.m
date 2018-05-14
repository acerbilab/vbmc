function y = infbench_goris2015(x,infprob,mcmc_params)
%INFBENCH_GORIS2015 Inference benchmark log pdf -- neuronal model from Goris et al. (2015).

if nargin < 3; mcmc_params = []; end

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
            if isempty(mcmc_params); id = 0; else; id = mcmc_params(1); end
            
            trinfo = infprob.Data.trinfo;
            
            if id == 0
            
                % First, check optimum            
                opts = optimoptions('fminunc','Display','off','MaxFunEvals',700);

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
                
            elseif id > 0 && n == mcmc_params(2)
                
                rng(id);
                widths = 0.5*(infprob.PUB - infprob.PLB);
                logpfun = @(x) infbench_goris2015(x,infprob);
                
                % Number of samples
                if numel(mcmc_params) > 2
                    W_mult = mcmc_params(3);
                else
                    W_mult = 200;
                end
                
                W = 2*(infprob.D+1);    % Number of walkers
                Ns = W*W_mult;             % Number of samples
                
                sampleopts.Burnin = Ns;
                sampleopts.Thin = 1;
                sampleopts.Display = 'iter';
                sampleopts.Diagnostics = false;
                sampleopts.VarTransform = false;
                sampleopts.InversionSample = false;
                sampleopts.FitGMM = false;
                sampleopts.TolX = 1e-5;
                % sampleopts.TransitionOperators = {'transSliceSampleRD'};

                x0 = xmin(infprob.idxParams);
                x0 = warpvars(x0,'d',trinfo);   % Convert to unconstrained coordinates
                [Xs,lls,exitflag,output] = eissample_lite(logpfun,x0,Ns,W,widths,infprob.LB,infprob.UB,sampleopts);
                
                filename = ['goris2015_mcmc_n' num2str(n) '_id' num2str(id) '.mat'];
                save(filename,'Xs','lls','exitflag','output');                
            end
            
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
        
        xmin = NaN(1,12);       fval = Inf;
        xmin_post = NaN(1,12);  fval_post = Inf;
        
        switch n
            case {1,2,3,4,5,6}; name = ['fake0', int2str(n)];
            case 7
                    name = 'm620r12';
                    xmin = [138.831269717139 2.36026404288357 0.646113173689136 1.191214614653 1 0.231291032974457 -0.272638945416596 3.09862498803914 72.8822298534178 0.00789002312857097 0.101380347749067 0.689049234550672];
                    fval = -2594.1103827372;
                    xmin_post = [138.843688065659 2.3665375005477 0.659599079735605 1.17814871632405 1 0.231000169669172 -0.272638945416596 3.09697483360828 72.8822298534178 0.00789002312857097 0.101380347749067 0.687089204130602];
                    fval_post = -2604.77539968962;
                    Mean_laplace = [-0.211867733531555 -0.0730665349668865 -0.275231596859294 -0.107383793519965 0.380017540695543 -0.0895586835326452 0.414845535843612];
                    Cov_laplace = [4.74613605583888e-05 1.99434355656865e-06 3.65689268922535e-05 -2.52065928710655e-05 5.04840689845336e-06 4.68400990698909e-06 -7.08334887161111e-07;1.99434355656865e-06 0.000497094401556914 0.000536330835889105 -0.000217552429691616 -4.713474662478e-05 -3.44392851542459e-05 -7.15221803997072e-06;3.65689268922535e-05 0.000536330835889105 0.00620962868333679 -0.00398289711494996 0.000156897367236504 0.000275821255748612 6.30110898039862e-06;-2.52065928710655e-05 -0.000217552429691616 -0.00398289711494996 0.0034304639973409 7.34903234836809e-05 -0.000265200011931196 -1.77092064664496e-05;5.04840689845336e-06 -4.713474662478e-05 0.000156897367236504 7.34903234836809e-05 0.000640814900416903 0.000742577093030104 -6.28275427525083e-06;4.68400990698909e-06 -3.44392851542459e-05 0.000275821255748612 -0.000265200011931196 0.000742577093030104 0.00113674536197343 -6.79207224232296e-06;-7.08334887161111e-07 -7.15221803997072e-06 6.30110898039862e-06 -1.77092064664496e-05 -6.28275427525083e-06 -6.79207224232296e-06 0.000383522189670278];
                    lnZ_laplace = -2620.215196872;                    
            case 8
                    name = 'm620r35';
                    xmin = [227.535999643729 3.01725050874219 2.4518747278566 0.86520211156348 0.886173951591468 -0.0401803925967166 -0.463062374186733 1.03064820242696 642.425443794774 0.001 8.65532672683422 0.0847961527858778];
                    fval = -6349.19310242853;
                    xmin_post = [227.525038766274 2.98441527013782 2.44110025601237 0.864158990877667 0.886173951591468 -0.0388962169756627 -0.463062374186733 1.04549627725566 642.425443794774 0.001 8.65532672683422 0.0849427245581592];
                    fval_post = -6365.94966510981;
                    Mean_laplace = [0.246159014730157 -0.00419042301523429 0.198121521016511 -0.199604945703338 -0.0628649200861632 -1.82097075638536 -0.0345884322911903];
                    Cov_laplace = [4.47822130862364e-06 4.06811675172822e-06 3.6839700263027e-06 -3.82675567403067e-06 4.53544306633322e-07 5.18510886971452e-06 8.55790792612653e-08;4.06811675172822e-06 0.000383801117843062 0.000383020465439865 -0.000303708816711443 1.92763075859813e-05 -0.000923605321289635 -6.61183740860532e-06;3.6839700263027e-06 0.000383020465439865 0.000667483336058314 -0.000451618013252342 3.42439543335783e-05 -0.00131115190375389 -5.99165957006139e-06;-3.82675567403067e-06 -0.000303708816711443 -0.000451618013252342 0.000347358260197232 -1.75606707913435e-05 0.000698307483272314 4.40444807854449e-06;4.53544306633322e-07 1.92763075859813e-05 3.42439543335783e-05 -1.75606707913435e-05 1.04207664592211e-05 4.98880653839811e-05 2.5874127818743e-07;5.18510886971452e-06 -0.000923605321289635 -0.00131115190375389 0.000698307483272314 4.98880653839811e-05 0.0124544523754372 5.93919515461926e-05;8.55790792612653e-08 -6.61183740860532e-06 -5.99165957006139e-06 4.40444807854449e-06 2.5874127818743e-07 5.93919515461926e-05 0.00016134204923219];
                    lnZ_laplace = -6381.87197047855;                    
%             case 7;             name = 'm620r12';
%                 xmin = [138.8505216444501 2.38317564009549 0.682321320237262 1.1613095607596 1 0.231748337632257 -0.272638945416596 3.10117864852662 72.8822298534178 0.00789002312857097 0.101380347749067 0.693895739234024];
%                 fval = -2594.08310420223;
%             case 8;             name = 'm620r35';
%                 xmin = [227.530092052922 3.00555244729356 2.44308608399358 0.867188243443111 0.886173951591468 -0.039623616648953 -0.463062374186733 1.03689776623743 642.425443794774 0.001 8.65532672683422 0.0851310168322118];
%                 fval = -6349.08642324277;
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
        y.Post.xBaseFull = xmin_post;
        xmin_post = warpvars(xmin_post(idx_params),'d',trinfo);
        fval_post = fval_post + warpvars(xmin_post,'logp',trinfo);
                
        y.Post.Mean = Mean_laplace;
        y.Post.Mode = xmin_post;          % Mode of the posterior
        y.Post.ModeFval = fval_post;        
        
        % range = 5*(y.PUB-y.PLB);
        y.Post.lnZ = lnZ_laplace;
        y.Post.Cov = Cov_laplace;
        
        % Save data and coordinate transformation struct
        y.Data = data;
        y.Data.trinfo = trinfo;
        
    end
    
else
    
    % Iteration call -- evaluate objective function
    
    % Transform unconstrained variables to original space
    x_orig = warpvars(x,'i',infprob.Data.trinfo);
    dy = warpvars(x,'logpdf',infprob.Data.trinfo);   % Jacobian correction
    
    if all(isfinite(infprob.Post.xBaseFull))
        xfull = infprob.Post.xBaseFull;
    else
        xfull = infprob.xBaseFull;        
    end
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