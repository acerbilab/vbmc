function [history,algo,algoset,flags,Nreadfiles,probstruct] = collectHistoryFiles(benchlist)

flags = 0;  % Overhead flag

if isempty(benchlist{4}); noise = [];
else; noise = ['@' benchlist{4} 'noise']; end

% Read data files
prob = benchlist{1};
subprob = benchlist{2};
basedir = [prob '@' subprob];
type = upper(benchlist{3});
subdir = [type noise];

% Check algorithm subtype
if ~isempty(benchlist{5})
    index = find(benchlist{5} == '@',1);
    if ~isempty(index)
        algo = benchlist{5}(1:index-1);
        algoset = benchlist{5}(index+1:end);
    else
        algo = benchlist{5};
        algoset = benchlist{6};
    end
    if isempty(algoset); algoset = 'base'; end
    algoset_file = algoset;
    if numel(algoset) > 9 && strcmp(algoset(end-8:end), '_overhead')
        algoset_file = algoset(1:end-9);
        flags(1) = 1;
    end
    basefilename = [algo '@' algoset_file '@*.mat'];
else
    algo = [];
    algoset = [];
    basefilename = '*.mat';
end

filesearch = dir([basedir filesep subdir filesep basefilename]);
Nfiles = numel(filesearch);
Nreadfiles = 0;

history = [];
if isempty(filesearch)
    algo = []; algoset = [];
    warning(['No files found for ''' [basedir filesep subdir filesep basefilename] ''' in path.']);
    return;
end

% Read history from each file
for iFile = 1:Nfiles
    filename = [basedir filesep subdir filesep filesearch(iFile).name];
    try
        temp = load(filename);
    catch
        temp = [];
        warning(['Error loading file ' filename '.']);
    end
    if ~isfield(temp,'history'); continue; end
    Nreadfiles = Nreadfiles + 1;
    if isfield(temp,'speedtest')
        t = temp.speedtest.start(:,1:4) + temp.speedtest.end(:,1:4);
        speedtest = sum(t(:));
    else
        speedtest = NaN;
    end    
    for i = 1:length(temp.history)
        temp.history{i}.speedtest = speedtest;
        history{end+1} = temp.history{i};
    end
end

% Also return associated probstruct
if nargout > 5
    clear infbench_func;
    probstruct = infprob_init(prob,subprob,type,[],1,[]);
end

end