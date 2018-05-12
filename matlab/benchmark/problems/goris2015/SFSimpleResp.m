function [M] = SFSimpleResp(varargin)

% SFSimpleResp       Computes response of simple cell for sf11 experiment
%
% SFSimpleResp(varargin) returns a simple cell response for simple sine 
% gratings that differ in spatial frequency. The cell's receptive field is 
% the n-th derivative of a 2-D Gaussian that need not be circularly 
% symmetric.

% Get input values from varargin or assign default values
S       = GetNamedInput(varargin, 'cellStructure', pwd);
channel = GetNamedInput(varargin, 'channel', pwd);
expRun  = GetNamedInput(varargin, 'expRun', pwd);

% Load the data structure
T = S.sf(expRun);

% Get preferred stimulus values
prefOr = pi/180 * channel.pref.or;                                         % in radians
prefSf = channel.pref.sf;                                                  % in cycles per degree
prefTf = round(nanmean(T.exp.trial.tf{1}));                                % in cycles per second

% Get directional selectivity
ds = channel.ds;

% Get derivative order in space and time
dOrdSp = channel.dord.sp;                                                  
dOrdTi = channel.dord.ti;                                                  

% Get aspect ratio in space
aRatSp = channel.arat.sp;

% Get spatial coordinates
xCo = 0;                                                                   % in visual degrees, centred on stimulus center
yCo = 0;                                                                   % in visual degrees, centred on stimulus center

% Store some results in M
M          = struct;
M.pref.or  = prefOr;
M.pref.sf  = prefSf;
M.pref.tf  = prefTf;
M.pref.xCo = xCo;
M.pref.yCo = yCo;
M.arat.sp  = aRatSp;
M.dord.sp  = dOrdSp;
M.dord.ti  = dOrdTi;
M.ds       = ds;

% Pre-allocate memory
z             = T.exp.trial;
nTrials       = numel(z.num);
nOr           = 1;
M.complexResp = zeros(120, nTrials, nOr);


% Compute simple cell response for all trials
for p = 1:nTrials
    
    % Set stim parameters
    for iC = 1:1
        stimOr(iC) = z.ori{iC}(p) * pi/180;                                % in radians
        stimTf(iC) = z.tf{iC}(p);                                          % in cycles per second
        stimCo(iC) = z.con{iC}(p);                                         % in Michelson contrast
        stimPh(iC) = z.ph{iC}(p) * pi/180;                                 % in radians
        stimSf(iC) = z.sf{iC}(p);                                          % in cycles per degree
    end
      
    
    % I. Orientation, spatial frequency and temporal frequency
    diffOr = repmat(prefOr, [1 1]) - repmat(stimOr', [1 nOr]);             % matrix size: 1 x nFilt (i.e., number of stimulus components by number of orientation filters)
    o      = (cos(diffOr).^2 .* exp(((aRatSp^2)-1) * cos(diffOr).^2)).^(dOrdSp/2);
    oMax   = exp(((aRatSp^2)-1)).^(dOrdSp/2);
    oNl    = o/oMax;
    e      = 1 + (ds*.5*(-1+(square(diffOr + pi/2))));
    selOr  = oNl.*e;

    if channel.dord.sp == 0
        selOr(:) = 1;
    end
    
    % Compute spatial frequency tuning
    sfRel = stimSf./prefSf;
    s     = stimSf.^dOrdSp .* exp(-dOrdSp/2 * sfRel.^2);
    sMax  = prefSf.^dOrdSp .* exp(-dOrdSp/2);
    sNl   = s/sMax;
    selSf = sNl;
    
    % Compute temporal frequency tuning
    tfRel = stimTf./prefTf;
    t     = stimTf.^dOrdTi .* exp(-dOrdTi/2 * tfRel.^2);
    tMax  = prefTf.^dOrdTi .* exp(-dOrdTi/2);
    tNl   = t'/tMax;
    selTf = tNl;    


    % II. Phase, space and time
    omegaX = stimSf.*cos(stimOr);                                          % the stimulus in frequency space
    omegaY = stimSf.*sin(stimOr);
    omegaT = stimTf;
    
    P(:,1) = 2*pi*repmat(xCo', [120 1]);                                   % P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
    P(:,2) = 2*pi*repmat(yCo', [120 1]);                                   % P(:,1) and p(:,2) describe location of the filters in space
    
    % Pre-allocate some variables
    respSimple = zeros(nOr, 120);
    
    for o = 1:nOr
        linR1 = zeros(120*length(xCo), 1);                                 % pre-allocation
        linR2 = zeros(120*length(xCo), 1);
        linR3 = zeros(120*length(xCo), 1);
        linR4 = zeros(120*length(xCo), 1);
        computeSum = 0;                                                    % important constant: if stimulus contrast or filter sensitivity equals zero there is no point in computing the response
        
        for c = 1:1                                                        % there are up to nine stimulus components
            selSi = selOr(c,o)*selSf(c)*selTf(c);                          % filter sensitivity for the sinusoid in the frequency domain
            
            if (selSi ~= 0 && stimCo(c) ~= 0)
                computeSum = 1;
                
                % Use the effective number of frames displayed/stimulus duration
                stimPos = (0:119)/120 + stimPh(c)/(2*pi*stimTf(c));        % 120 frames + the appropriate phase-offset
                P3Temp  = (repmat(stimPos, [length(xCo) 1]));
                P(:,3)  = 2*pi*P3Temp(:);                                  % P(:,3) describes relative location of the filters in time.
                
                rComplex = selSi*stimCo(c)*exp(1i*P*[omegaX(c) omegaY(c) omegaT(c)]');
                
                linR1(:,c) = real(rComplex);                               % four filters placed in quadrature
                linR2(:,c) = -1*real(rComplex);
                linR3(:,c) = imag(rComplex);
                linR4(:,c) = -1*imag(rComplex);
            end
        end
        
        if computeSum == 1
            respSimple1 = max(0, sum(linR1, 2));                           % superposition and half-wave rectification,...
            respSimple2 = max(0, sum(linR2, 2));
            respSimple3 = max(0, sum(linR3, 2));
            respSimple4 = max(0, sum(linR4, 2));                         
            
            %  if channel is tuned, it is phase selective...
            if channel.dord.sp ~= 0
                respSimple(o,:) = respSimple1;
            elseif channel.dord.sp == 0
                respComplex = (respSimple1.^2 + respSimple2.^2 + respSimple3.^2 + respSimple4.^2);
                respSimple(o,:) = sqrt(respComplex);
            end
        end
    end

    % Store response in desired format
    M.simpleResp(:,p) = reshape(respSimple', [120 1]);
end
end





%% Functions used in the main script
%% GetNamedInput
function y = GetNamedInput(C, varName, varDefault)
% looks for the string varName in varargin, and returns the following entry
% in varargin. If varName is named more than once, a cell array is
% returned. If it is not found, varDefault is returned.

y = varDefault;

k = 0;
for i = 1:(length(C)-1)
    if strcmpi(C{i}, varName)
        k = k+1;                                                           % increment k every time the varName is found in varargin
        if k > 1
            y{k} = C{i+1};
        else
            y = C{i+1};
        end
    end
end
end
%%

