function benchdata = loadSummaryStats(fileName,benchlist,layers,dimlayers)

% Read summary statistics from benchmark file if present
if exist(fileName,'file')
    load(fileName,'benchdata');

    for iLayer = 1:numel(layers)
        benchlist{dimlayers} = layers{iLayer};
        index = find(benchlist{5} == '@',1);
        if ~isempty(index)
            algo = benchlist{5}(1:index-1);
            algoset = benchlist{5}(index+1:end);
        else
            algo = benchlist{5};
            algoset = benchlist{6};
        end
        if isempty(algoset); algoset = 'base'; end
        if isempty(benchlist{4}); noise = [];
        else noise = ['_' benchlist{4} 'noise']; end

        field1 = ['f1_' benchlist{1} '_' benchlist{2}];
        field2 = ['f2_' upper(benchlist{3}) noise];
        field3 = ['f3_' algo '_' algoset];

        % Summary statistics
        try
            if Noisy
                storedMinBag = benchdata.(field1).(field2).MinBag;
%                storedMinBag = benchdata.(field1).(field2).(field3).MinBag;
                MinBag.fval = storedMinBag.fval(:);
                MinBag.fsd = storedMinBag.fsd(:);
            else
                MinFval = min(MinFval, ...
                    benchdata.(field1).(field2).(field3).MinFval);
            end
        catch
            % Field not present, just skip
        end
    end
else
    benchdata = [];
end
