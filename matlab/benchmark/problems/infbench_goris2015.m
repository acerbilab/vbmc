function y = infbench_goris2015(x,infprob,mcmc_params)
%INFBENCH_GORIS2015 Inference benchmark log pdf -- neuronal model from Goris et al. (2015).

if nargin < 3; mcmc_params = []; end

if isempty(x)
    if isempty(infprob) % Generate this document        
        fprintf('\n');

%        for n = 7:12
        for n = 11
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
                logpfun = @(x) -nlogpost(x,infprob);
                
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
                LB = infprob.PLB - 10*widths;
                UB = infprob.PUB + 10*widths;
                
                [Xs,lls,exitflag,output] = eissample_lite(logpfun,x0,Ns,W,widths,LB,UB,sampleopts);
                
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
        Mean_laplace = NaN(1,7);    Cov_laplace = NaN(7,7); lnZ_laplace = NaN;
        Mean_mcmc = NaN(1,7);       Cov_mcmc = NaN(7,7);    lnZ_mcmc = NaN;
        
        switch n
            case {1,2,3,4,5,6}; name = ['fake0', int2str(n)];
            case 7
                    name = 'm620r12';
                    xmin = [138.831269717139 2.36026404288357 0.646113173689136 1.191214614653 1 0.231291032974457 -0.272638945416596 3.09862498803914 72.8822298534178 0.00789002312857097 0.101380347749067 0.689049234550672];
                    fval = -2594.1103827372;
                    xmin_post = [138.843688065659 2.3665375005477 0.659599079735605 1.17814871632405 1 0.231000169669172 -0.272638945416596 3.09697483360828 72.8822298534178 0.00789002312857097 0.101380347749067 0.687089204130602];
                    fval_post = -2604.77539968962;
                    Mean_laplace = [-0.465520991260055 -1.69627543340411 -1.62447989201296 -0.767119415827957 0.470491515145298 -0.484169040138803 -2.60814931957517];
                    Cov_laplace = [0.000229133759218794 1.82326721309945e-05 0.000410425618637872 -0.000282902243846033 1.37333646747451e-05 2.55742161512486e-05 -7.49219827856342e-06;1.82326721309945e-05 0.00860578855990285 0.0113987299239016 -0.0046236785662946 -0.000242809250998626 -0.000356073871350522 -0.000143256002177701;0.000410425618637872 0.0113987299239016 0.162017238465403 -0.103918933733934 0.000992228291153627 0.00350094756853489 0.000154939078874722;-0.000282902243846033 -0.0046236785662946 -0.103918933733934 0.089505239660401 0.000464757180894073 -0.00336613410893957 -0.000435454797885497;1.37333646747451e-05 -0.000242809250998626 0.000992228291153627 0.000464757180894073 0.000982265536813824 0.00228454532591384 -3.74450494783302e-05;2.55742161512486e-05 -0.000356073871350522 0.00350094756853489 -0.00336613410893957 0.00228454532591384 0.00701913099415825 -8.12472287376342e-05;-7.49219827856342e-06 -0.000143256002177701 0.000154939078874722 -0.000435454797885497 -3.74450494783301e-05 -8.12472287376342e-05 0.00888756642652754];
                    lnZ_laplace = -2620.215196872;
                    % R_max = 1.011. Ntot = 400000. Neff_min = 8013.5. Total funccount = 3124820.
                    Mean_mcmc = [-0.465802680530785 -1.70566976804436 -5.19522208334754 -0.384915835875687 0.467083855425486 -0.496086661684798 -2.58834415856342];
                    Cov_mcmc = [0.000224749978866004 5.60815708807962e-06 0.00017447753573741 3.34220973948264e-05 9.77989454620153e-06 1.19603450094093e-05 -3.31163521355875e-06;5.60815708807962e-06 0.00941413340619495 0.0427992417936741 6.60756302336674e-05 -0.000214807115205535 -0.000450661540007981 -1.85983014361911e-05;0.00017447753573741 0.0427992417936741 8.88269095379091 -0.49970931844157 0.00488012844427025 0.0134563621915982 -0.0103060212088849;3.34220973948264e-05 6.60756302336674e-05 -0.49970931844157 0.0977994680732417 0.000643845852213431 -0.00338833139986662 0.000243456644557692;9.77989454620153e-06 -0.000214807115205535 0.00488012844427025 0.000643845852213431 0.000954197299194386 0.00219688034751615 -9.24820491816269e-07;1.19603450094093e-05 -0.000450661540007981 0.0134563621915982 -0.00338833139986662 0.00219688034751615 0.00675804037157995 1.9303310108393e-05;-3.31163521355875e-06 -1.85983014361911e-05 -0.0103060212088849 0.000243456644557692 -9.24820491816269e-07 1.9303310108393e-05 0.00869582976950535];
                    lnZ_mcmc = -2619.18872742075;
            case 8
                    name = 'm620r35';
                    xmin = [227.535999643729 3.01725050874219 2.4518747278566 0.86520211156348 0.886173951591468 -0.0401803925967166 -0.463062374186733 1.03064820242696 642.425443794774 0.001 8.65532672683422 0.0847961527858778];
                    fval = -6349.19310242853;
                    xmin_post = [227.525038766274 2.98441527013782 2.44110025601237 0.864158990877667 0.886173951591468 -0.0388962169756627 -0.463062374186733 1.04549627725566 642.425443794774 0.001 8.65532672683422 0.0849427245581592];
                    fval_post = -6365.94966510981;
                    Mean_laplace = [0.540866637097969 -1.40969632775648 0.793390622057475 -1.23818170784033 -0.0778317007333179 -4.78656641349373 -4.77167510180345];
                    Cov_laplace = [2.16199382189283e-05 3.71915052853202e-05 4.13464601119852e-05 -4.29489924469928e-05 1.23379305282585e-06 2.83101653572452e-05 9.05186857154558e-07;3.71915052853202e-05 0.00664443465640919 0.00814039870304761 -0.00645477482468913 9.92996916739682e-05 -0.00954931906625238 -0.000132432399140008;4.13464601119852e-05 0.00814039870304761 0.0174155029784697 -0.0117832976945504 0.000216560805889953 -0.0166422056812428 -0.00014732997280385;-4.29489924469928e-05 -0.00645477482468913 -0.0117832976945504 0.0090630259787226 -0.000111054727543906 0.00886348616975374 0.000108301749797391;1.23379305282585e-06 9.92996916739682e-05 0.000216560805889953 -0.000111054727543906 1.59733485495094e-05 0.000153481096658657 1.54209118156668e-06;2.83101653572452e-05 -0.00954931906625238 -0.0166422056812428 0.00886348616975374 0.000153481096658657 0.0769032675285656 0.000710450551803561;9.05186857154558e-07 -0.000132432399140008 -0.00014732997280385 0.000108301749797391 1.54209118156668e-06 0.000710450551803561 0.00373886627309871];
                    lnZ_laplace = -6381.87197047855;
                    % R_max = 1.006. Ntot = 392000. Neff_min = 380138.9. Total funccount = 2643186.
                    Mean_mcmc = [0.540822611749478 -1.41890717619093 0.797316705837751 -1.23656853487087 -0.0784821116557497 -4.86881498861325 -4.7659191271868];
                    Cov_mcmc = [2.15014902722377e-05 2.27645972680799e-06 -7.57507403377745e-06 6.36186602122456e-06 -1.92021541848178e-07 -2.44185363429748e-05 7.14013274337131e-07;2.27645972680799e-06 0.00138815068547392 -0.000696012920741727 0.000419745080381005 1.29435286349826e-05 -0.00245659284030882 -4.79065239795959e-05;-7.57507403377745e-06 -0.000696012920741727 0.00864652224310871 -0.00551714221253578 7.71900807551871e-06 -0.00260724977995793 4.4470396090958e-05;6.36186602122456e-06 0.000419745080381005 -0.00551714221253578 0.00462659836349837 6.82804060704023e-05 -0.00174461377936581 -1.77460724999311e-05;-1.92021541848178e-07 1.29435286349826e-05 7.71900807551871e-06 6.82804060704023e-05 2.3939543579508e-05 0.000610075482119828 7.03179447276722e-06;-2.44185363429748e-05 -0.00245659284030882 -0.00260724977995793 -0.00174461377936581 0.000610075482119828 0.0956555910880346 0.000695495102318744;7.14013274337131e-07 -4.79065239795959e-05 4.4470396090958e-05 -1.77460724999311e-05 7.03179447276722e-06 0.000695495102318744 0.00373535266638093];
                    lnZ_mcmc = -6382.21082591018;
            case 9;             name = 'm624l54';
                xmin = [219.1171252067330 1.39227769925336 3.03322024598022 0.617344863953975 0.355100712841601 0.067816661803776 -0.12988300841421 1.22284211173093 345.53039808358 0.001 4.29514870404523 0.308744430060413];
                fval = -6497.67358979187;
            case 10;            name = 'm625r58';    
                xmin = [305.847145521642 4.7748407736943 1.65588982473794 3.5 0.0559001728606333 0.0448467266543214 -0.192486743740829 4.81143259854959 349.659335938905 0.0700397166151502 0.213318145774832 1.05373239451993];
                fval = -2118.86479506553;                
			case 11
				name = 'm625r62';
				xmin = [124.88710608225 3.68586576122773 2.96330657248823 0.328529880587874 0.00572852306727856 -0.547736645287558 0.410971814338923 4.56809152611831 59810475.5470703 0.001 0.001 0.129162534142113];
				fval = -302.033482141767;
				xmin_post = [124.865607867591 3.68556099680715 2.92014194714368 0.365291069910745 0.00572852306727856 -0.511617271295265 0.410971814338923 4.69464968545128 59810475.5470703 0.001 0.001 0.129337540160696];
				fval_post = -314.509056865964;
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
                
        y.Post.Mean = Mean_mcmc;
        y.Post.Mode = xmin_post;          % Mode of the posterior
        y.Post.ModeFval = fval_post;        
        y.Post.lnZ = lnZ_mcmc;
        y.Post.Cov = Cov_mcmc;
        
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