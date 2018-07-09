function [XStars,YMean,YSD,GP_y,monitor]=track_sbq(XsFull,YsFull,...
    GP_y,params,training_set_size)


if nargin<4 || ~isfield(params,'maxpts')
    params.maxpts=1000; % max number of data points to store
end
if ~isfield(params,'lookahead')
    params.lookahead=0; % how many steps to lookahead
end
if ~isfield(params,'threshold')
    params.threshold=1.e-3; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'deldrop')
    params.deldrop=1; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'step')
    params.step=max(1,params.lookahead); % how fine-grained should predictions be made - predictions will be made every params.step up to lookahead
end
if ~isfield(params,'print')
    params.print=1;%not reassuring dots;
end
if ~isfield(params,'training_set_size')
    % the number of points from XsFull and YsFull that the GP starts with
    params.training_set_size=1;
end

if size(XsFull,2)==1
    XsFull = allcombs({1,XsFull});
end

want_monitor = nargout>4;

% Initialises hypersamples
GP_y = sample_hyperparameters(GP_y);

NDims=length(unique(XsFull(:,1)));
NData=size(XsFull,1);
times=unique(XsFull(:,2));
delt=min(times(2:end)-times(1:end-1));

lowr.LT=true;
uppr.UT=true;

XData=XsFull(1:params.training_set_size,:);
YData=YsFull(1:params.training_set_size,:);

GP_y=revise_GP(XData, YData, GP_y, 'overwrite', []);

XStars=[];
YMean=[];
YSD=[];
dropped=0;

count=0;

hp_functions = [r, tilde_r, eps_r, q, tilde_q, eps_q, q', q'', tilde_q'', eps_q'']; % all the various functions of hyperparameters that we maintain GPs over.
for f = hp_functions
     [GP_f] = initialise_quad_GP(GP_y) 
     % sets up (Gaussian) covariance functions, quadrature-hypersamples, etc.
end

for ind=training_set_size+1:NData
    
    
    if params.print==1 && ind>0.01*(1+count)*NData
        count=count+1;
        ind
    elseif params.print==0 && ( rem(ind,100) == 0)
        fprintf('.');
    end
        
	X = XsFull(ind, :);
	Y = YsFull(ind, :);
    
    % nans in our data are ignored
    null_data = any(isnan([X,Y]),2);
    X(null_data,:) = [];
    Y(null_data,:) = [];
    
    dropped = max([dropped, length(YData) - (params.maxpts - size(X,1))]);
    % The -size(X,1) is because we're just about to add additional pts on

    if (dropped > 0)
        GP_y = revise_GP(XData, YData, GP_y, 'downdate', 1:dropped);
        XData(1:dropped, :) = [];
        YData(1:dropped ,:) = [];
    end

    new_data_inds = length(YData)+(1:length(Y))';
    XData=[XData;X];
    YData=[YData;Y]; 

    GP_y=revise_GP(XData, YData, GP_y, 'update', new_data_inds);
    % this update will also overwrite moved hypersamples
    
    
    % nexttime is the time at which we will next get a reading
    if ind==NData
        nexttime=XsFull(ind,2)+delt;
    else
        nexttime=XsFull(ind+1,2);
    end
    
	% XStar is the point at which we wish to make predictions. Note XStar
	% will be empty if we are about to receive more observations at the
	% same time.
    XStar=allcombs({(1:NDims)',...
        (max(X(:,2))+params.lookahead*delt:...
        params.step*delt:...
        nexttime+(params.lookahead)*delt)'});
    XStars=[XStars;XStar];
    
    GP_y = make_predictions(XStar,[],GP_y);
    
    for f = hp_functions
        f_samples = as appropriate;
        f_posterior = update_posterior(GP_y.hypersample(:).hyperparams, f_samples,
        GP_f);
    end

    if ~isempty(ystar)
       predictions = bz_quad_hps(GP_q, GP_eps_q, GP_r, GP_eps_r);
       % predictions is just a vector
    end
    mean = bz_quad_hps(GP_q', 0, GP_r, GP_eps_r);
    second_moment = bz_quad_hps(GP_q'', GP_eps_q'', GP_r, GP_eps_r);
    sd = sqrt(second_moment - mean^2);

    
    YMean=[YMean;YStar_mean];
    YSD=[YSD;YStar_sd];

    GP_y = sbq(GP_q', GP_r, GP_tilde_r, GP_eps_r);
    GP_y = set_candidates(GP_y); %sets up the new GP_y.candidates

   
    if want_monitor
        samples=cat(1,GP_y.hypersample.hyperparameters);
        num_samples=size(samples,1);
        monitor.t(ind).rho=zeros(num_samples,1);
        monitor.t(ind).rho(closestInd)=1;
        monitor.t(ind).hypersample=cat(1,GP_y.hypersample.hyperparameters);
    end
    
end