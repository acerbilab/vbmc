function [vp0_vec,vp0_type,elcbo_beta,compute_var,NSentK,NSentKFast] = vpsieve_vbmc(Ninit,Nbest,vp,gp,optimState,options,K)
%VPSIEVE Preliminary 'sieve' method for fitting variational posterior.

% Assign default values to OPTIMSTATE
if ~isfield(optimState,'delta'); optimState.delta = 0; end
if ~isfield(optimState,'EntropySwitch'); optimState.EntropySwitch = false; end
if ~isfield(optimState,'Warmup'); optimState.Warmup = ~vp.optimize_weights; end
if ~isfield(optimState,'temperature'); optimState.temperature = 1; end
if ~isfield(optimState,'Neff'); optimState.Neff = size(gp.X,1); end

if isempty(Nbest); Nbest = 1; end
if nargin < 7 || isempty(K); K = vp.K; end

%% Set up optimization variables and options

vp.delta = optimState.delta(:);

if isempty(Ninit) % Number of initial starting points
    Ninit = ceil(evaloption_vbmc(options.NSelbo,K));
end
nelcbo_fill = zeros(Ninit,1);

% Number of samples per component for MC approximation of the entropy
NSentK = ceil(evaloption_vbmc(options.NSent,K)/K);

% Number of samples per component for preliminary MC approximation of the entropy
NSentKFast = ceil(evaloption_vbmc(options.NSentFast,K)/K);

% Deterministic entropy if entropy switch is on or only one component
if optimState.EntropySwitch || K == 1
    NSentK = 0;
    NSentKFast = 0;
end

% Confidence weight
elcbo_beta = evaloption_vbmc(options.ELCBOWeight,optimState.Neff);
compute_var = elcbo_beta ~= 0;

% Compute soft bounds for variational parameters optimization
[vp,thetabnd] = vpbounds(vp,gp,options,K);

%% Perform quick shotgun evaluation of many candidate parameters

if Ninit > 0
    % Get high-posterior density points
    [Xstar,ystar] = gethpd_vbmc(gp.X,gp.y,options.HPDFrac);

    % Generate a bunch of random candidate variational parameters
    switch Nbest
        case 1
            [vp0_vec,vp0_type] = vbinit_vbmc(1,Ninit,vp,K,Xstar,ystar);
        otherwise
            [vp0_vec1,vp0_type1] = vbinit_vbmc(1,ceil(Ninit/3),vp,K,Xstar,ystar);
            [vp0_vec2,vp0_type2] = vbinit_vbmc(2,ceil(Ninit/3),vp,K,Xstar,ystar);
            [vp0_vec3,vp0_type3] = vbinit_vbmc(3,Ninit-2*ceil(Ninit/3),vp,K,Xstar,ystar);
            vp0_vec = [vp0_vec1,vp0_vec2,vp0_vec3];
            vp0_type = [vp0_type1;vp0_type2;vp0_type3];
    end
    
    if isfield(optimState,'vp_repo') && ~isempty(optimState.vp_repo) && options.VariationalInitRepo
        Ntheta = numel(get_vptheta(vp0_vec(1)));
        idx = find(cellfun(@numel,optimState.vp_repo) == Ntheta);
        if ~isempty(idx)
            vp0_vec4 = [];
            for ii = 1:numel(idx)
                vp0_vec4 = [vp0_vec4,rescale_params(vp0_vec(1),optimState.vp_repo{idx(ii)})];
            end
            vp0_vec = [vp0_vec,vp0_vec4];
            vp0_type = [vp0_type;ones(numel(vp0_vec4),1)];        
        end
    end

    % Quickly estimate ELCBO at each candidate variational posterior
    for iOpt = 1:numel(vp0_vec)
        [theta0,vp0_vec(iOpt)] = get_vptheta(vp0_vec(iOpt),vp.optimize_mu,vp.optimize_sigma,vp.optimize_lambda,vp.optimize_weights);        
        [nelbo_tmp,~,~,~,varF_tmp] = negelcbo_vbmc(theta0,0,vp0_vec(iOpt),gp,NSentKFast,0,compute_var,options.AltMCEntropy,thetabnd);
        nelcbo_fill(iOpt) = nelbo_tmp + elcbo_beta*sqrt(varF_tmp);
    end

    % Sort by negative ELCBO
    [~,vp0_ord] = sort(nelcbo_fill,'ascend');
    vp0_vec = vp0_vec(vp0_ord);
    vp0_type = vp0_type(vp0_ord);    
else
    vp0_vec = vp;
    vp0_type = 1;
end



end