% TPGIVEBOF   Computes the negative log likelihood for the LN-LN model

function [NLL, respModel] = TPGiveBof(params, structureTP, structureSF)

MIN_P = 1e-10;      % Small correction to avoid zero probabilities

% Get parameter values
% Excitatory channel
excChannel.pref.or = params(1);
excChannel.pref.sf = params(2);
excChannel.arat.sp = params(3);
excChannel.dord.sp = params(4);
excChannel.dord.ti = 0.25;                                                 % derivative order in the temporal domain, d = 0.25 ensures broad tuning for temporal frequency
excChannel.ds      = params(5);

% Inhibitory channel
inhChannel.pref.or = 0;
inhChannel.pref.sf = excChannel.pref.sf;
inhChannel.arat.sp = 1;
inhChannel.dord.sp = 0;
inhChannel.dord.ti = excChannel.dord.ti;
inhChannel.ds      = 0;
inhChannel.gain    = params(6);

% Other (nonlinear) model components
sigma    = 10^params(7);                                                   % normalization constant
respExp  = params(8);                                                      % response exponent
scale    = params(9);                                                      % response scalar

% Noise parameters
noiseEarly = params(10);                                                   % early additive noise
noiseLate  = params(11);                                                   % late additive noise
varGain    = params(12);                                                   % multiplicative noise


%% Evaluate prior on response exponent -- corresponds loosely to the measurements in Priebe et al. (2004)
priorExp = lognpdf(respExp, 1.15, 0.3);
NLLExp   = -log(priorExp);

%% Evaluate texpat experiment
for iR = 1:numel(structureTP.tp)
    T = structureTP.tp(iR);

    % Get simple cell response for excitatory and inhibitory channel
    [E] = TPSimpleResp('cellStructure', structureTP, 'channel', excChannel, 'expRun', iR);  
    [I] = TPSimpleResp('cellStructure', structureTP, 'channel', inhChannel, 'expRun', iR);  

    % Extract simple cell responses (half-rectified linear filtering)
    Lexc = E.simpleResp;
    Linh = I.simpleResp;

    % Compute full model response (the normalization pool consists of many cells)
    numerator        = noiseEarly + Lexc + inhChannel.gain*Linh;
    denominator      = sigma.^2 + T.mod.normalization.normResp';
    ratio            = max(0, numerator./denominator).^respExp;
    meanRate         = mean(ratio);
    respModel.tp{iR} = noiseLate + scale*meanRate;
    
    % Get predicted spike count distributions
    mu  = max(.01, T.exp.trial.duration.*respModel.tp{iR});                % The predicted mean spike count
    var = mu + (varGain*(mu.^2));                                          % The corresponding variance of the spike count
    r   = (mu.^2)./(var - mu);                                             % The parameters r and p of the negative binomial distribution
    p   = r./(r + mu);
    
    % Evaluate the model
    llh = nbinpdf(T.exp.trial.spikeCount, r, p);                           % The likelihood for each pass under the doubly stochastic model
    
    llh = MIN_P + (1-MIN_P)*llh;        % Correct for zeroes
    
    NLLtempTP(iR) = sum(-log(llh));                                        % The negative log-likelihood of the whole data-set      
end



%% Evaluate sf11 experiment
for iR = 1:numel(structureSF.sf)
    T = structureSF.sf(iR);

    % Get simple cell response for excitatory and inhibitory channel
    [E] = SFSimpleResp('cellStructure', structureSF, 'channel', excChannel, 'expRun', iR);  
    [I] = SFSimpleResp('cellStructure', structureSF, 'channel', inhChannel, 'expRun', iR);  

    % Extract simple cell responses (half-rectified linear filtering)
    Lexc = E.simpleResp;
    Linh = I.simpleResp;

    % Get normalization response (given full contrast images: simply 1)
    normResp = ones(size(Lexc));

    % Compute full model response
    numerator        = noiseEarly + Lexc + inhChannel.gain*Linh;
    denominator      = sigma.^2 + normResp;
    ratio            = max(0, numerator./denominator).^respExp;
    meanRate         = mean(ratio);
    respModel.sf{iR} = noiseLate + scale*meanRate;
    
    % Get predicted spike count distributions
    mu  = max(.01, T.exp.trial.duration.*respModel.sf{iR});                % The predicted mean spike count
    var = mu + (varGain*(mu.^2));                                          % The corresponding variance of the spike count
    r   = (mu.^2)./(var - mu);                                             % The parameters r and p of the negative binomial distribution
    p   = r./(r + mu);
        
    % Evaluate the model
    llh = nbinpdf(T.exp.trial.spikeCount, r, p);                           % The likelihood for each pass under the doubly stochastic model
        
    llh = MIN_P + (1-MIN_P)*llh;        % Correct for zeroes
    
    NLLtempSF(iR) = sum(-log(llh));                                        % The negative log-likelihood of the whole data-set      
end


%% Combine data sets and prior
NLL = sum(NLLtempTP) + sum(NLLtempSF) + NLLExp;
