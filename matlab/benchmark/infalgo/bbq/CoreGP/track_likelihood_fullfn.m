function covvy = track_likelihood_fullfn(XData,YData,covvy,inds,flag)

NHyperparams = numel(covvy.hyperparams);
NData = size(XData,1);

if nargin<5
    flag = 'overwrite';
end

if isfield(covvy.hypersamples(1),'logL')
    logLs = [covvy.hypersamples.logL];
    min_logL = min(logLs);
else
    min_logL = -inf;
end

for ind = inds
%     try
         covvy = gpparams(XData,YData,covvy,flag,[],ind);
%     catch ME
%         % Sometimes a hypersample is moved into such a position so as to
%         % generate errors in the likelihood function. The prototypical
%         % example is when an input scale hyperparameter becomes so large
%         % that it generates conditioning errors in the GP likelihood model.
%         % In such a case, we simply assign a zero likelihood (and zero to
%         % its gradient).
%         
%         covvy.hypersamples(ind).logL = min_logL;
%         covvy.hypersamples(ind).glogL = mat2cell(zeros(NHyperparams,1),ones(NHyperparams,1),1);
% 
%         covvy.hypersamples(ind).datahalf = zeros(NData,1);
%         covvy.hypersamples(ind).datatwothirds = zeros(NData,1);
%         covvy.hypersamples(ind).cholK = eye(NData);
%         
%         disp(getReport(ME))
%         disp(['Likelihood for sample ',num2str(ind),' set to zero.']);
%     end
end