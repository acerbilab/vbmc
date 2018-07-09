function [probstruct,history] = infbench_run(probset,prob,subprob,noise,algo,idlist,options)
%INFBENCH_RUN Run posterior and model inference benchmark for expensive likelihoods.

% Luigi Acerbi 2018

clear functions;

if nargin < 7; options = []; end

% Convert string input back to numeric arrays
prob = inputstr2num(prob);
subprob = inputstr2num(subprob);
noise = inputstr2num(noise);
idlist = inputstr2num(idlist);

if ischar(options); options = eval(options); end

% Get default options
defopts = infbench_defaults('options');

% Assign default values to OPTIONS struct
for f = fieldnames(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

charsep = options.CharFileSep;

% Root of the benchmark directory tree
if isempty(options.RootDirectory)
    options.RootDirectory = fileparts(mfilename('fullpath'));
end

% Matlab file path
if isempty(options.PathDirectory)
    options.PathDirectory = fileparts(mfilename('fullpath'));
end

% Add sub-directories to path
folders = {'gplite','infalgo','problems'};
for iFolder = 1:numel(folders)
    addpath([options.PathDirectory filesep() folders{iFolder}]);
end
% addpath(genpath(options.PathDirectory));
if ~isempty(options.ProblemDirectory)
    addpath(genpath(options.ProblemDirectory));    
end

% Inference algorithm settings
setidx = find(algo == charsep,1);
if isempty(setidx)
    algoset = 'base';
else
    algoset = algo(setidx+1:end);    
    algo = algo(1:setidx-1);
end
options.Algorithm = algo;
options.AlgorithmSetup = algoset;

% Noise setting (lo-me-hi, or nothing)
if isempty(noise); noisestring = []; else; noisestring = [charsep noise 'noise']; end

scratch_flag = false;
timeOffset = 0; % Computation time that does not count for benchmark

% Test processor speed (for baseline)
speedtest = [];
if options.SpeedTests > 0; speedtest.start = bench(options.SpeedTests); end

% Loop over inference runs
for iRun = 1:length(idlist)
    clear infbench_func;    % Clear persistent variables
    
    % Initialize random number generator to current id
    rng(idlist(iRun),'twister');

    % Initialize current problem
    probstruct = infprob_init(probset,prob,subprob,noise,idlist(iRun),options);
    
    % Create working dir
    directoryname = [probstruct.ProbSet charsep probstruct.Prob];
    subdirectoryname = [probstruct.SubProb noisestring];
    mkdir([options.RootDirectory filesep directoryname]);
    mkdir([options.RootDirectory filesep directoryname filesep subdirectoryname]);
    
    % Copy local data file to working dir
    if isfield(probstruct,'LocalDataFile') && ~isempty(probstruct.LocalDataFile)
        targetfile = [options.RootDirectory filesep directoryname filesep subdirectoryname filesep probstruct.LocalDataFile];
        if ~exist(targetfile,'file')
            disp(['Copying data file ' probstruct.LocalDataFile ' to local folder.']);
            copyfile([ '.' filesep probstruct.LocalDataFile],targetfile);
        else
            disp(['Data file ' probstruct.LocalDataFile ' already exists in local folder.']);
        end
    end
    
    % Move to working dir
    cd([options.RootDirectory filesep directoryname filesep subdirectoryname]);    
    
    probstruct.nIters = 0;
    FirstPoint = [];    % First starting point of the run
    
    % Loop until out of budget of function evaluations
    probstruct.nIters = probstruct.nIters + 1;

    % Update iteration counter
    infbench_func([],probstruct,1,probstruct.nIters,timeOffset);

    % Starting point (used only by some algorithms)
    probstruct.InitPoint = [];
    if probstruct.StartFromMode
        if any(isnan(probstruct.Mode))
            warning('Cannot start from mode, vector contains NaNs. Setting a random starting point.');
        else
            probstruct.InitPoint = probstruct.Mode;
        end
    end
    if isempty(probstruct.InitPoint)
        probstruct.InitPoint = rand(1,probstruct.D).*(probstruct.PUB-probstruct.PLB) + probstruct.PLB;
    end
    if isempty(FirstPoint); FirstPoint = probstruct.InitPoint; end

    % Run inference
    algofun = str2func(['infalgo_' algo]);
    probstruct.AlgoTimer = tic;
    [history{iRun},post,algoptions] = algofun(algo,algoset,probstruct);
    
    history{iRun}.X0 = FirstPoint;
    history{iRun}.Algorithm = algo;
    history{iRun}.AlgoSetup = algoset;
    history{iRun}.Output.post = post;
    history{iRun}.lnZ_true = probstruct.lnZ;
    history{iRun}.lnZpost_true = probstruct.Post.lnZ;    
    if isfield(history{iRun},'scratch'); scratch_flag = true; end
end

% Test processor speed (for baseline)
if options.SpeedTests > 0; speedtest.end = bench(options.SpeedTests); end

% Save inference results
filename = [options.OutputDataPrefix algo charsep algoset charsep num2str(idlist(1)) '.mat'];
if scratch_flag  % Remove scratch field from saved files, keep it for output
    temp = history;
    for iRun = 1:numel(history)
        history{iRun} = rmfield(history{iRun},'scratch');
    end
    save(filename,'history','speedtest');
    history = temp;
    clear temp;
else
    save(filename,'history','speedtest');    
end

% Save algorithm options
filename = [options.OutputDataPrefix algo charsep algoset charsep 'opts.mat'];
if ~exist(filename,'file'); save(filename,'algoptions'); end

cd(options.RootDirectory); % Go back to root

%--------------------------------------------------------------------------
function s = inputstr2num(s)
%INPUTSTR2NUM Converts an input string into a numerical array (if needed).

if ischar(s)
    % Allowed chars in numerical array
    allowedchars = '0123456789.-[]() ,;:';
    isallowed = any(bsxfun(@eq, s, allowedchars'),1);
    if strcmp(s,'[]')
        s = [];
    elseif all(isallowed)
        t = str2num(s);
        if ~isempty(t) && isnumeric(t) && isvector(t); s = t; end
    end
end