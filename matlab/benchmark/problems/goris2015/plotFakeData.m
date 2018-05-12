% plottFakeData    Plots fake data analogous to the fake data described in
%       the supplement of Goris, Simoncelli & Movshon (2015)
function plotFakeData()
%%

%% Begin with clean slate
clear all
close all
clc

%% Set paths
currentPath  = '.'; %strcat('/Users/goris/Desktop/Luigi/Scripts');
functionPath = '.'; %strcat('/Users/goris/Desktop/Luigi/Functions');
loadPath     = '.'; %strcat('/Users/goris/Desktop/Luigi/Structures');

%% Set constants
nFakeCells = 6;

%% Loop through data
for iF = 1:nFakeCells
    
    % Go to data path
    cd(loadPath)
    
    % Set loadnames
    loadNameSf = strcat('fake0', int2str(iF), '_sf');
    loadNameTp = strcat('fake0', int2str(iF), '_tp');
    
    % Load the spatial frequency data
    load(loadNameSf);
    sSF = S;
    
    % Load the orientation mixture data
    load(loadNameTp);
    sTP = S;
    
    % Get the ground truth parameters of the LN-LN model (13 in total)
    groundTruth(:,iF) = sTP.paramsGen;

    % Go to function path
    cd(functionPath)
    
    % Set starting points
    startvalues = [120 3.5 1 1 0.5 -0.1 0 3 4e3 0.1 0.1 0.1];
    
    % The parameters and their bounds
    % 01 = preferred direction of motion (degrees), unbounded, logical to use most effective stimulus value for family 1, high contrast as starting point
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
    
    % Get the negative log likelihood of the data under this parameterization
    [NLL(iF), respModel{iF}] = TPGiveBof(startvalues, sTP, sSF);     
    % NLL is the negative log likelihood
    % respModel{iF}.sf{1} contains the trial-by-trial predicted spike count for the spatial frequency measurements
    % respModel{iF}.tp{1} contains the trial-by-trial predicted spike count for the orientation mixtures
    
    % Summarize the simulated responses for the spatial frequency measurements
    blockList = unique(sSF.sf.exp.trial.blockID);
    
    for iSF = 1:numel(blockList)-2
        indCond         = find(sSF.sf.exp.trial.blockID == iSF);
        stimSf(iSF)     = unique(sSF.sf.exp.trial.sf{1}(indCond));
        rateSf{iF}(iSF) = nanmean(sSF.sf.exp.trial.spikeCount(indCond));
    end
    
    % Summarize the predicted responses for the orientation mixture measurements
    for iE = 1:2
        for iW = 1:5
            StimBlockIDs  = ((iE-1)*5+iW-1)*16:((iE-1)*5+iW-1)*16+16-1;
            nStimBlockIDs = length(StimBlockIDs);
            
            % Average model prediction across trials
            rate{iW}{iE} = nan(1, nStimBlockIDs);
            iC = 0;
            for iB = StimBlockIDs
                indCond = find(sTP.tp.exp.trial.blockID == iB);
                if ~isempty(indCond)
                    iC               = iC+1;
                    rate{iW}{iE}(iC) = mean(sTP.tp.exp.trial.spikeCount(indCond));
                end
            end
            
            % Store data in structured fashion
            oriCentered  = sTP.tp(1).exp.ori{iW}{iE} - sTP.paramsGen(1);   % Center on preferred orientation (for plotting purposes)
            plotMat(:,1) = 180 * sawtooth(pi/180 * oriCentered + pi);
            plotMat(:,2) = rate{iW}{iE};
            pMat{iW}{iE} = sortrows(plotMat, 1);
            
            rateOm{iF}(iW,iE,:) = pMat{iW}{iE}([1:end, 1], 2);
        end
    end
end
cd(currentPath)



%% Make figure
% Set colors, one for each fake cell
col{1} = [0 0 0];
col{2} = [1 0 0];
col{3} = [1 .5 0];
col{4} = [0  1 .25];
col{5} = [0 .5 1];
col{6} = [.5 0 1];

% Set stimulus orientation axis
stimOri = [pMat{1}{1}(:,1); pMat{1}{1}(1,1)+360];

% Create figure 1
set(figure(1), 'OuterPosition', [200 200 2000 1200])
for iF = 1:nFakeCells
    panelNum = ((iF-1)*11) + 1;
    
    subplot(6,11,panelNum)
    semilogx(stimSf, rateSf{iF}, '-', 'color', col{iF}, 'linewidth', 2)
    box off, axis square
    axis([.1 10 0 120])
    ylabel('Firing rate (ips)')
    
    if iF == 1
        title('Family 1, high contrast')
    end
    
    if iF == nFakeCells
        xlabel('Spatial frequency (c/deg)')
    end
    
    for iW = 1:5
        for iE = 1:2
            panelNum = ((iF-1)*11) + 1 + ((iE-1)*5) + iW;
            
            subplot(6,11,panelNum)
            plot(stimOri, squeeze(rateOm{iF}(iW,iE,:)), '-', 'color', col{iF}, 'linewidth', 2)
            box off, axis square
            axis([-200 200 0 120])
            
            if iF == 1
                if iE == 1
                    title(sprintf('Family %d, high contrast', iW))
                else
                    title(sprintf('Family %d, low contrast', iW))
                end
            end
            
            if iF == nFakeCells
                xlabel('Orientation (deg)')
            end
        end
    end
end
cd(currentPath)


