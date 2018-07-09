function [XStars YMean YSD covvy closestInd monitor] = track_BQ(XsFull, YsFull, covvy, lookahead, params,training_set_size)

NDims = length(unique(XsFull(:, 1)));
NData = size(XsFull, 1);
% times=unique(XsFull(:,2));
% delt=min(times(2:end)-times(1:end-1));

if NData == 0 || NDims == 0
    XStars =[];
    YMean =[];
    YSD =[]; 
    closestInd=1;
    monitor=[];
    return
end

if (nargin < 4)
    lookahead = 0; % how many steps to lookahead
end

if (nargin < 5 || ~isfield(params, 'maxpts'))
    params.maxpts = 1000; % max number of data points to store
end
if (~isfield(params, 'threshold'))
    params.threshold = 1.e-3; % thresh for desired accuracy [determines the # data points to store]
end
if (~isfield(params, 'deldrop'))
    params.deldrop = 1; % thresh for desired accuracy [determines the # data points to store]
end
if (~isfield(params, 'step'))
    params.step = 1;%max(1,lookahead); % how fine-grained should predictions be made - predictions will be made every params.step up to lookahead
end
if (~isfield(params, 'print'))
    params.print = 0; %reassuring dots;
end

% draws samples and assigns parameters via bayes MCMC
if ~isfield(covvy, 'hypersamples')
    covvy = hyperparams(covvy);
end
covvy = bmcparams(covvy);


lowr.LT = true;
uppr.UT = true;

XData=XsFull([],:);
YData=YsFull([],:);


XStars = [];
YMean = [];
YSD = [];
dropped = 0;

count = 0;

for ind = 1:NData
    
	if (params.print == 1 && ind > 0.01 * (1 + count) * NData)
		count = count + 1;
		ind
	elseif (params.print == 0 && (rem(ind, 100) == 0))
		fprintf('.');
	end
	
	X = XsFull(ind, :);
	Y = YsFull(ind, :);
    
    dropped = max([dropped, length(YData) - (params.maxpts - size(X,1))]);
	% The -size(X,1) is because we're just about to add additional pts on
	
	if (dropped > 0)
        XData(1:dropped, :) = [];
		YData(1:dropped ,:) = [];
		covvy = gpparams(XData, YData, covvy, 'downdate', 1:dropped);
	end
	
	
	XData = [XData; X];
	YData = [YData; Y]; 
	
    if ind > 1 
        covvy = gpparams(XData, YData, covvy, 'update', length(YData));
    else
        covvy = gpparams(XData, YData, covvy, 'overwrite');
    end
	rho = weights(covvy);
	
% 	% nexttime is the time at which we will next get a reading
% 	if (ind == NData)
% 		nexttime = XsFull(ind,2) + delt;
% 	else
% 		nexttime = XsFull(ind + 1, 2);
% 	end
% 	
% % 	XStar is the point at which we wish to make predictions. Note XStar
% % 	will be empty if we are about to receive more observations at the
% % 	same time.
% 	XStar = allcombs({(1:NDims)', (max(X(:,2)) + lookahead * delt: ...
%                      params.step * delt:nexttime + ...
%                      (lookahead) * delt)'});

    if (ind + lookahead) > NData
		XStar = XsFull([], :);
	else
		XStar = XsFull(ind + lookahead, :);
	end

	XStars = [XStars; XStar];
	
	[wm wC] = weighted_gpmeancov(rho, XStar, XData, covvy);
	
	wsig = sqrt(wC);
	
	YMean = [YMean; wm];
	YSD = [YSD; wsig];
	
	[m, closestInd] = max(rho);
	
% 	[K] = covvy.covfn(covvy.hypersamples(closestInd).hyperparameters);
% 	Kstst = K(XStar, XStar);
% 	KStarData = K(XStar, XData);
% 	test_sig = wsig;
% 	drop = params.deldrop;
% 	cholK = covvy.hypersamples(closestInd).cholK;
% 	while (~isempty(test_sig) && min(test_sig < params.threshold) && drop < size(XData, 1))
% 		cholK = downdatechol(cholK, 1:params.deldrop); 
% 		KStarData = KStarData(:, 1 + params.deldrop:end);
% 		Khalf = linsolve(cholK', KStarData', lowr)';
% 		test_sig = min(sqrt(diag(Kstst - Khalf * Khalf')));    
% 		drop = drop + params.deldrop;
% 	end
% 	dropped = drop - params.deldrop;
%     
        monitor.t(ind).rho=rho;
         monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
end