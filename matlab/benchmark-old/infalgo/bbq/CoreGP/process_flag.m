function [flag, grad_hp_inds] = process_flag(flag)

if iscell(flag)
    grad_hp_inds = flag{2};
    flag = flag{1};
else
    grad_hp_inds = [];
end