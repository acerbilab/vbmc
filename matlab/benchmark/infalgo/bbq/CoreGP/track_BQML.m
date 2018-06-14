function [XsFullData, YsFullData, XStars, YMean, YSD, covvy, monitor]...
    = track_BQML(XsFull,YsFull,covvy,params)

want_monitor = nargout>6;

NData=size(XsFull,1);

if nargin<4
    params = struct();
end
if  ~isfield(params,'lookahead')
    params.lookahead = 1;
end
lookahead=params.lookahead; % how many steps to lookahead
if ~isfield(params,'maxpts')
    params.maxpts=500; 
    % max number of data points to store
end
if ~isfield(params,'threshold')
    params.threshold=1.e-3; 
    % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'deldrop')
    params.deldrop=1; 
    % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'step')
    params.step=max(1,lookahead); 
    % how fine-grained should predictions be made - predictions will be
    % made every params.step up to lookahead
end
if ~isfield(params,'print')
    params.print=0;
    %not reassuring dots;
end
if ~isfield(params,'hypersample_shift_period')
    params.hypersample_shift_period = ceil(NData/10);
end
if ~isfield(params,'training_set')
    % the indices from XsFull and YsFull that the GP starts with
    training_set=[];
else
    training_set=params.training_set;
end

XTrainData=XsFull(training_set,:);
YTrainData=YsFull(training_set,:);

if isempty(covvy)
    if isfield(params,'num_hypersamples')
        num_hypersamples = params.num_hypersamples;
    else
        num_hypersamples = 100;
    end
    if isfield(params,'covfn_name')
        covfn_name = params.covfn_name;
    else
        covfn_name = {'matern',5/2};
    end    
    if isfield(params,'meanfn_name')
        meanfn_name = params.meanfn_name;
    else
        meanfn_name = 'constant';
    end

    covvy = set_covvy(covfn_name, meanfn_name, covvy, ...
        XTrainData, YTrainData, num_hypersamples);
end

% draws samples and assigns parameters via bayes MCMC
if ~isfield(covvy, 'hypersamples')
    covvy = hyperparams(covvy);
end
active_inds = covvy.active_hp_inds;


NSamples = numel(covvy.hypersamples);
% All hypersamples need to have their gpparams overwritten
covvy.lastHyperSampleMoved=1:NSamples;



lowr.LT=true;
uppr.UT=true;





if ~isempty(YTrainData)
    % All hypersamples need to have their gpparams overwritten, otherwise
    % we do not do any further training.
    
    if ~isfield(params,'num_optim_steps')
        % the number of steps taken during the optimisation procedure.
        num_optim_steps=0;
    else
        num_optim_steps=params.num_optim_steps;
    end
    
    covvy=gpparams(XTrainData, YTrainData, covvy, 'overwrite', []);
    if num_optim_steps>0 && ~isempty(active_inds)
        [dummyVar,closestInd] = max([covvy.hypersamples.logL]);

        priorMeans=[covvy.hyperparams(active_inds).priorMean];
        priorSDs=[covvy.hyperparams(active_inds).priorSD];
        lower_bound = priorMeans - 3*priorSDs;
        upper_bound = priorMeans + 3*priorSDs;

        starting_pt = covvy.hypersamples(closestInd).hyperparameters(active_inds);

        options = optimset('GradObj','on','MaxFunEvals',num_optim_steps,...
            'Display','iter');

        
        display('Beginning optimisation of hyperparameters')
        tic;
 

        objective = @(a_hypersample) neg_log_likelihood(a_hypersample,...
            active_inds,XTrainData,YTrainData,covvy,params);

        [best_a_hypersample,dummyVar,dummyVar,output] = fmincon(objective,...
            starting_pt,...
            [],[],[],[],...
            lower_bound, upper_bound,...
            [],options);
        
        display('Completed optimisation of hyperparameters')
        toc;

        covvy.hypersamples(1).hyperparameters(active_inds) = best_a_hypersample;
        covvy = gpparams(XTrainData,YTrainData,covvy,'overwrite',[],1);
        
        if want_monitor
            monitor.output = output;
        end

        best_hypersample = covvy.hypersamples(1).hyperparameters;
        num_hps = numel(covvy.hyperparams);

        for hp = 1:num_hps
            gp.hyperparams(hp).samples = [];
            gp.hyperparams(hp).priorMean = best_hypersample(hp);
        end
        
        covvy = hyperparams(covvy);
        
    end
end

covvy = bmcparams(covvy);
covvy.lastHyperSampleMoved=[];

XStars=[];
YMean=[];
YSD=[];
XData = [];
YData = [];


test_indices = setdiff(1:NData, training_set);
XsFullData = XsFull(test_indices,:);
YsFullData = YsFull(test_indices,:);

display('Begun prediction');
tic

count=0;
for ind=1:length(test_indices)-lookahead
    
    
    if params.print==1 && ind>0.01*(1+count)*NData
        count=count+1;
        ind
    elseif params.print==0 && ( rem(ind,100) == 0)
        fprintf('.');
    end
    
    move_hypersamples_now = mod(ind,params.hypersample_shift_period) == 0;
    
    if move_hypersamples_now 
                
        covvy.derivs_cov = true;
        covvy.derivs_mean = true;
    else
        
        covvy.derivs_cov = false;
        covvy.derivs_mean = false;
    end
        
	X = XsFullData(ind, :);
	Y = YsFullData(ind, :);
    
    no_data = any(isnan([X,Y]));
    
    if ~no_data
        dropped = length(YData) - (params.maxpts - size(X,1));
        % The -size(X,1) is because we're just about to add additional pts on

        if (dropped > 0)
            XData(1:dropped, :) = [];
            YData(1:dropped ,:) = [];
            covvy = gpparams(XData, YData, covvy, 'downdate', 1:dropped);
        end

        XData=[XData;X];
        YData=[YData;Y]; 
        
        if ind==1
            covvy=gpparams(XData, YData, covvy, 'overwrite');
        else
            covvy=gpparams(XData, YData, covvy, 'update', size(XData,1), ...
                setdiff(1:NSamples,covvy.lastHyperSampleMoved));
            covvy=gpparams(XData, YData, covvy, 'overwrite', [], ...
                covvy.lastHyperSampleMoved);
        end
    end
    
 	rho = weights(covvy);

    if (ind == NData)
		XStar = XsFullData([], :);
	else
		XStar = XsFullData(ind + lookahead, :);
	end

	XStars = [XStars; XStar];
	
	[wm wC] = weighted_gpmeancov(rho, XStar, XData, covvy);
    
    wsig = sqrt(wC);
	
	YMean = [YMean; wm];
	YSD = [YSD; wsig];
	
%     if move_hypersamples_now
%         covvy = manage_hyper_samples_ML(covvy,'all');
%         covvy = bq_params(covvy);
%     else
%         covvy.lastHyperSampleMoved=[];
%     end
   
    samples=cat(1,covvy.hypersamples.hyperparameters);
    num_samples=size(samples,1);
    if want_monitor
        monitor.t(ind).rho=zeros(num_samples,1);
        monitor.t(ind).rho(closestInd)=1;
        monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
    end
end

XsFullData = XsFullData(1:end-lookahead,:);
YsFullData = YsFullData(1:end-lookahead,:);

display('Completed prediction');
toc

function [neg_logL neg_glogL] = neg_log_likelihood(a_hypersample,...
    active_inds,XData,YData,covvy,params)


want_derivs = nargout>1;
covvy.use_derivatives = want_derivs;

if size(a_hypersample,1)>1
    covvy.hypersamples(1).hyperparameters(active_inds) = a_hypersample';
else
    covvy.hypersamples(1).hyperparameters(active_inds) = a_hypersample;
end

covvy = gpparams(XData,YData,covvy,'overwrite',[],1);
neg_logL = -covvy.hypersamples(1).logL;
if want_derivs
neg_glogL = -[covvy.hypersamples(1).glogL{active_inds}];
end
if params.print
    fprintf('.');
end