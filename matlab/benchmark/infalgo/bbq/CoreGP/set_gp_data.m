function [gp, flag] = ...
    set_gp_data(gp, X_data, y_data, flag, active)

if nargin<5
    active = length(y_data);
end
if nargin<4
    flag = 'overwrite';
end



[NData,NDims] = size(X_data);
switch flag
    case 'update'
        if NData == length(active)
            X_new = X_data;
            y_new = y_data;
        else
            X_new = X_data(active,:);
            y_new = y_data(active,:);
        end
        
        gp.X_data(active,:) = X_new;
        gp.y_data(active,:) = y_new;
        
        [NData,NDims] = size(gp.X_data);
               
        if gp.sqd_diffs_cov
            old_sqd_diffs_data = gp.sqd_diffs_data;
            NOldData = size(old_sqd_diffs_data,1);
            gp.sqd_diffs_data = nan(NData,NData,NDims);
            gp.sqd_diffs_data(1:NOldData,1:NOldData,:) = old_sqd_diffs_data;
 
            new_sqd_diffs = sqd_diffs(gp.X_data, X_new);
            gp.sqd_diffs_data(:, active, :) = new_sqd_diffs;
            gp.sqd_diffs_data(active, :, :) = tr(new_sqd_diffs);
        end
        
    case 'downdate'
        gp.X_data(active,:) = [];
        gp.y_data(active) = [];
        
        if gp.sqd_diffs_cov
            gp.sqd_diffs_data(active,:,:) = [];
            gp.sqd_diffs_data(:,active,:) = [];
        end
    case 'overwrite'
        gp.X_data = X_data;
        gp.y_data = y_data;
        
        if gp.sqd_diffs_cov
            gp.sqd_diffs_data = nan(NData,NData,NDims);
            gp.sqd_diffs_data = sqd_diffs(X_data, X_data);
        end
    case 'new_hps'
        flag = 'overwrite';
        % but don't change data
end

function D = sqd_diffs(X, Y)

[num_r_X, num_dims] = size(X);
[num_r_Y, num_dims] = size(Y);

X = reshape(X, num_r_X, 1, num_dims);
trY = reshape(Y, 1, num_r_Y, num_dims);

D = abs(bsxfun(@minus, X, trY)).^2;