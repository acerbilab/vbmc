function parlist = infbench_joblist(filename,checkdir,probset,prob,subprob,noise,algo,algoset,idlist)
%INFBENCH_JOBLIST Write to file list of benchmarks factorially combined.

benchdef = infbench_defaults('options');        % Get default parameters
charsep=benchdef.CharFileSep;

% Build combination out of provided factors
parlist=combcell(probset,prob,subprob,noise,algo,algoset,idlist);
for i=1:length(parlist)
 
end

% Write to file
fout=fopen(filename,'w+');
for i=1:length(parlist)
    noise = parlist{i}{4};
    if isempty(noise) || strcmpi(noise,'[]')
        noisestring = [];
    else
        noisestring = [charsep noise 'noise'];
    end
    directoryname = [parlist{i}{1} charsep parlist{i}{2}];
    subdirectoryname = [parlist{i}{3} noisestring];
    if ischar(parlist{i}{7})
        idlist = str2num(parlist{i}{7});
    else
        idlist = parlist{i}{7};
    end
    if any(parlist{i}{5} == charsep)
        algo = parlist{i}{5};
    else
        if strcmp(parlist{i}{6},'[]'); parlist{i}{6} = 'base'; end
        algo = [parlist{i}{5} charsep parlist{i}{6}];
    end
    
    filename = [algo charsep num2str(idlist(1)) '.mat'];

    % Check if file already exist; write line if it doesn't
    checkname = [checkdir filesep directoryname filesep subdirectoryname filesep filename];
    doWrite = isempty(checkdir) | ~exist(checkname,'file');
    if doWrite
        fprintf(fout,['''%s'',''%s'',''%s'',''%s'',''%s'',''%s''\n'], ...
            parlist{i}{1},parlist{i}{2},parlist{i}{3},parlist{i}{4},algo,parlist{i}{7});
    end
end
fclose(fout);
