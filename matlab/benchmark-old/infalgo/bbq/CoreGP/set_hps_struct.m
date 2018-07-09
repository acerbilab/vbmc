function hps_struct = set_hps_struct(covvy,num_dims)
% Creates a structure hps_struct containing the positions of each
% hyperparameter

if isfield(covvy,'hyperparams')
    num_hps = numel(covvy.hyperparams);

    names = {covvy.hyperparams.name};
    posns_vec = 1:num_hps;
    posns = num2cell(posns_vec);

    joint = [names;posns];
    hps_struct = struct(joint{:});
    
    hps_struct.num_hps = num_hps;
    
    is_mean_cell = strfind(names,'Mean');
    mean_inds = find(~cellfun(@(x) isempty(x),is_mean_cell));
    hps_struct.mean_inds = mean_inds;

    is_input_scale_cell = strfind(names,'logInputScale');
    input_scale_inds = find(~cellfun(@(x) isempty(x),is_input_scale_cell));

    if ~isempty(input_scale_inds)
        hps_struct.logInputScales = input_scale_inds;
    end
    
    %names={'PeriodInput1', 'Doover','LadidaInput123','badger67','InputScale2', 'InputScale1', 'PeriodInput3','Blah'}

    input_patterns = {'Input','Period','log_w0'};
    input_inds = [];
    for i = 1:length(input_patterns)
        input_inds = [input_inds, ...
            find(~cellfun(@(x) isempty(x),strfind(names,input_patterns{i})))];        
    end
    input_inds = unique(input_inds);
    input_names = names(input_inds);
    
    nums = regexp(input_names,'\d*','match');
    nums = nums(~cellfun(@isempty, nums));
    nums = cellfun(@(x) str2num(x{1}), nums);
    
    
    
    if  nargin<2 && isfield(covvy,'num_dims')
        num_dims = covvy.num_dims;
    elseif nargin<2
        num_dims = max(nums);
    end
    hps_struct.num_dims = num_dims;
    
    hps_struct.input_inds = cell(num_dims, 1);
    
    [unique_nums, dummy, unique_inds] = unique(nums);
    for i = 1:length(unique_nums)
        hps_struct.input_inds{unique_nums(i)} = ...
            input_inds(unique_inds==i);
    end
    
%         is_planar_weight_cell = strfind(names,'PlanarMeanWeight');
%     planar_weight_inds = find(~cellfun(@(x) isempty(x),is_planar_weight_cell));
% 
%     hps_struct.PlanarMeanWeights = planar_weight_inds;
% 
%     is_quad_weight_cell = strfind(names,'QuadMeanWeight');
%     quad_weight_inds = find(~cellfun(@(x) isempty(x),is_quad_weight_cell));
% 
%     hps_struct.QuadMeanWeights = quad_weight_inds;
%     
    is_multiple_hp = strfind(names,'1');
    multiple_hp_inds = find(~cellfun(@(x) isempty(x),is_multiple_hp));
    
    for ind = 1:length(multiple_hp_inds)
        hp = multiple_hp_inds(ind);
        name = names{hp};
        name = name(1:(is_multiple_hp{hp}-1));
        is_hp = strfind(names,name);
        hp_inds = find(~cellfun(@(x) isempty(x),is_hp));
        hps_struct.([name,'s']) = hp_inds;
    end
    
    
else
    hps_struct = struct([]);
end