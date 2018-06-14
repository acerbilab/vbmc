function noise_fun = get_noise(gp, flag)

if nargin<2
    flag = 'plain';
end

noise_fun_test = isfield(gp,'noisefn');
if ~noise_fun_test 
    if (~isfield(gp, 'logNoiseSDPos'))
        names = {gp.hyperparams.name};
        logNoiseSDPos = cellfun(@(x) strcmpi(x, 'logNoiseSD'), names);
    else
        logNoiseSDPos = gp.logNoiseSDPos;
    end
    
    % if logNoiseSDPos is empty, this means we assume zero noise.
    if ~isempty(logNoiseSDPos)
        noise_fun = iid_noise_fn(logNoiseSDPos, flag);
    else
        noise_fun = @(X,hps) 0; % no matter the flag
    end
else
    noise_fun = gp.noisefn(flag);
end
