function [gp] = ...
    revise_spgp(X_data, y_data, gp, flag, active, grad_hp_inds)
% [gp] = ...
%     revise_spgp(X_data, y_data, gp, flag, active, grad_hp_inds)
% sparse equivalent of revise_gp
%  gp = revise_spgp(X_data, y_data, gp, flag, active, samples, hps)
% X_data:   (additional) evaluated inputs to condition gp on
% y_data:   (additional) evaluated outputs to condition gp on
% gp:       existing gp structure (can be empty)
% flag:     if flag=='overwrite', we ignore what is in gp and just
%           calculate and store all terms afresh (see below for further)
% active:   the indices of evaluated data that are either new, if 
%           flag=='update', or to be removed, if flag=='downdate'. If
%           flag=='update', X_data and y_data can be either of full length,
%           with the new entries in positions indicated by active, or,
%           alternatively, of length exactly equal to that of active,
%           containing only new entries.
% grad_hp_inds: we compute the derivative of the log likelihood with respect to
%           these hyperparameters.

forced_sqd_jitter = 10^-10;
allowed_error = 10^-10;

% samples:  the hyperparameter samples to revise. 
samples = 1:numel(gp.hypersamples);

if (nargin < 4)
	flag = 'overwrite';
end

flag = lower(flag);

if (nargin < 5) || isempty(active)
    switch flag
        case 'update'
            % assume we have added data to the end of our existing
            % y_data and X_data
            
            existing_length = length(gp.hypersamples(samples(1)).inv_gamma_d);
            active = existing_length + (1:length(y_data)); 
        case {'overwrite'}
            active = [];
    end
end

grad_hyperparams = nargin > 5 && ~isempty(grad_hp_inds);

diag_sqd_noise = gp.diag_sqd_noise;
if grad_hyperparams
    Ddiag_sqd_noise = get_diag_sqd_noise(gp, {'grad hyperparams',grad_hp_inds});
end

Mu = gp.Mu;
% assume mean has zero derivative with respect to active hyperparameters
% for now.

gp.sqd_diffs_cov = false;
[gp, flag] = ...
    set_gp_data(gp, X_data, y_data, flag, active);
y_data = gp.y_data;
X_data = gp.X_data;
[num_data,num_dims] = size(X_data);

num_c = size(gp.hypersamples(1).X_c,1);

if isfield(gp, 'lambda_ind')
    lambda_ind = gp.lambda_ind;
    % global input scales
    w0_inds = gp.w0_inds;
else
    hps_struct = set_hps_struct(gp);
    lambda_ind = hps_struct.log_lambda;
    % global input scales
    w0_inds = hps_struct.log_w0s;
    %noise_ind = hps_struct.logNoiseSD;
end

zeroth_X_c_ind = numel(gp.hyperparams);
X_c_inds = zeroth_X_c_ind + ...
            (1:(num_dims*num_c));
w_c_inds = max(X_c_inds) + (1:(num_dims*num_c));

if isempty(y_data) || isempty(X_data)
    return
end

updating = strcmp(flag, 'update');
if updating
    if max(active) ~= num_data
        error('revise_spgp assumes that data are added in at the end of existing matrices');
    end
    num_active = length(active);
    X_active = X_data(active,:);
    y_active = y_data(active,:);
end
overwriting = strcmp(flag, 'overwrite');

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;


if num_data < num_dims
    warning('revise_spgp:small_num_data', 'make sure each data point is a row of X_data');
end

temp_hypersamples = gp.hypersamples;

names = {'y_data_minus_Mu', 'X_c', 'R_c', 'G_cd', 'inv_gamma_d', 'Ups_cd', ...
    'M_c', 'h_c', 'T_cd', 'A_cd', 'logL', 'glogL'};

%best_logL = -inf;



for sample_ind = 1:length(samples)
    
    sample = samples(sample_ind);
    
    hs = temp_hypersamples(sample).hyperparameters;
    if ~all(isreal(hs))
        warning('complex hyperparameter sample');
    end
    lambda = exp(hs(lambda_ind));
    w_0 = exp(hs(w0_inds));
    n_d = lambda^2/(sqrt(prod(2*pi*w_0)));
    
    % centres
    X_c = temp_hypersamples(sample).X_c;
    tw_c = exp(temp_hypersamples(sample).log_tw_c);
    w_c = bsxfun(@plus, tw_c, 0.5*w_0);
    
    if overwriting
    
        w_stack_c0 = bsxfun(@plus,...
                reshape(tw_c,num_c,1,num_dims),...
                reshape(tw_c,1,num_c,num_dims));

        dist_stack_c = ...
            bsxfun(@minus,...
                    reshape(X_c,num_c,1,num_dims),...
                    reshape(X_c,1,num_c,num_dims));

        sqd_dist_stack_c = dist_stack_c.^2;
        
        dist_stack_cd = ...
            bsxfun(@minus,...
                reshape(X_c,num_c,1,num_dims),...
                reshape(X_data,1,num_data,num_dims));

        sqd_dist_stack_cd = dist_stack_cd.^2;

        w_stack_cd = repmat(...
        reshape(w_c,num_c,1,num_dims),...
                1,num_data,1);

        N_cd_inv_lambda = 1./(sqrt(prod(2*pi.*w_stack_cd,3))).*...
                exp(-0.5*sum(sqd_dist_stack_cd./w_stack_cd,3));
            
        N_cd = lambda * N_cd_inv_lambda;
        clear N_cd_inv_lambda;
        
        N_c = 1./(sqrt(prod(2*pi.*w_stack_c0,3))).*...
            exp(-0.5*sum(sqd_dist_stack_c./w_stack_c0,3));
        
        [N_c, N_cd] = ...
            improve_X_c_problems(allowed_error, N_c, N_cd);

        while allowed_error > 0
            try
                R_c = chol(N_c);
                break
            catch
                warning('revise_spgp:X_c_problems', 'elements of X_c may be too close together')
                allowed_error = allowed_error/10;
                
                [N_c, N_cd] = ...
                    improve_X_c_problems(allowed_error, N_c, N_cd);
             end
        end

        



        
        y_data_minus_Mu = y_data - Mu(hs,X_data);

    
  
        
        G_cd = linsolve(R_c, N_cd, lowr);
        T_cd = linsolve(R_c, G_cd, uppr);
        
        
        % row vector of length num_data
        sqd_noise_d = max(forced_sqd_jitter, diag_sqd_noise(hs, X_data));

        gamma_d = sqd_noise_d + ...
                    max(0, n_d - sum(G_cd.^2,1));
        inv_gamma_d = gamma_d.^(-1);
        
        Ups_cd = bsxfun(@times, G_cd, inv_gamma_d);
        
        M_c = eye(num_c) + Ups_cd*G_cd';
        
        M_c = improve_covariance_conditioning(M_c, [], allowed_error);
        while allowed_error > 0
            try
                S_c = chol(M_c);
                break
            catch
                warning('revise_spgp:X_c_problems', 'elements of X_c may be too close together')
                allowed_error = allowed_error/10;
                M_c = improve_covariance_conditioning(M_c, [], allowed_error);
            end
        end
        
        
        A_cd = linsolve(S_c, Ups_cd, lowr);
        
        h_c = A_cd * y_data_minus_Mu;
    
	elseif updating
        
        % hyperparameters (including X_c and w_c) remain unchanged, but we
        % have added in a new point to the end of X_d.
        
        w_stack_ca = repmat(...
            reshape(w_c,num_c,1,num_dims),...
                    1,num_active,1);

        sqd_dist_stack_ca = ...
            bsxfun(@minus,...
                    reshape(X_c,num_c,1,num_dims),...
                    reshape(X_active,1,num_active,num_dims)).^2;

        N_ca_inv_lambda = 1./(sqrt(prod(2*pi.*w_stack_ca,3))).*...
                exp(-0.5*sum(sqd_dist_stack_ca./w_stack_ca,3));
        
        % new row
        y_data_minus_Mu_old = temp_hypersamples(sample).y_data_minus_Mu;
        y_active_minus_Mu = y_active - Mu(hs,X_active);
        y_data_minus_Mu = [y_data_minus_Mu_old; y_active_minus_Mu];
		
        % unchanged
        R_c = temp_hypersamples(sample).R_c;
        
        % new columns
        old_G_cd = temp_hypersamples(sample).G_cd;
        
        N_ca = lambda * N_ca_inv_lambda;
        clear N_cd_inv_lambda;
                
        G_ca = linsolve(R_c, N_ca, lowr);
        G_cd = [old_G_cd, G_ca];
        
        % has to be recalculated from scratch
        T_cd = linsolve(R_c, G_cd, uppr);
               
        % new diagonal elems
        old_inv_gamma_d = temp_hypersamples(sample).inv_gamma_d;
        
        sqd_noise_a = max(forced_sqd_jitter, diag_sqd_noise(hs, X_active));
        
        gamma_a = sqd_noise_a ...
             + max(0,n_d - sum(G_ca.^2,1));        
         inv_gamma_a = gamma_a.^(-1);
        
        inv_gamma_d = [old_inv_gamma_d,inv_gamma_a];
        
        % new columns
        Ups_cd_old = temp_hypersamples(sample).Ups_cd;
        
        Ups_ca = bsxfun(@times, G_ca, inv_gamma_a);
        Ups_cd = [Ups_cd_old,Ups_ca];
        
        % can be efficiently updated
        old_M_c = temp_hypersamples(sample).M_c;

        % has to be recalculated from scratch
        M_c = old_M_c + Ups_ca*G_ca';
        
        M_c = improve_covariance_conditioning(M_c, [], allowed_error);
        while allowed_error > 0
            try
                S_c = chol(M_c);
                break
            catch
                %warning('elements of X_c may be too close together')
                allowed_error = allowed_error/10;
                M_c = improve_covariance_conditioning(M_c, [], allowed_error);
            end
        end       
        
        % has to be recalculated from scratch
        A_cd = linsolve(S_c, Ups_cd, lowr);

        % has to be recalculated from scratch
        h_c = A_cd * y_data_minus_Mu;
               
    end
    
    
    h_d = inv_gamma_d' .* y_data_minus_Mu;
	
    logL = - 0.5 * num_data * log(2*pi)...
            - 0.5 * sum(-log(inv_gamma_d)) ...
            - sum(log(diag(S_c))) ...
            - 0.5 * y_data_minus_Mu' * h_d...
            + 0.5 * (h_c' * h_c);
        
%     if logL > best_logL
%         best_logL = logL;
%         best_N_c = N_c;
%         best_N_cd = N_cd;
%     end

    if grad_hyperparams
                        
        num_derivs = length(grad_hp_inds);

        g_c = T_cd * h_d;
        C_c = T_cd * A_cd';

        Dn_d = Dn_d_fn(grad_hp_inds, num_data, num_derivs, ...
            n_d, lambda_ind, w0_inds);
        DN_c = DN_c_fn(grad_hp_inds, ...
            num_c, num_dims, num_derivs, ...
            N_c, tw_c, ...
            w_stack_c0, dist_stack_c, sqd_dist_stack_c, ...
            X_c_inds, w_c_inds);
        DN_cd = DN_cd_fn(grad_hp_inds, ...
            num_c, num_data, num_dims, num_derivs, ...
            N_cd, w_0, tw_c, ...
            w_stack_cd, dist_stack_cd, sqd_dist_stack_cd, ...
            lambda_ind, w0_inds, X_c_inds, w_c_inds);


        % noise will always be first hyperparam, mean 
        Dsqd_noise_d = Ddiag_sqd_noise(hs, X_data);

        % assume for now the mean is not a hyperparameter
        %DMucell = DMu(hs, X_data);

        %stacks, each (:,:,i) representing the derivative wrt the ith
        %hyperparam
        T_dc_DN_cd = sum(bsxfun(@times, T_cd, DN_cd),1);
        T_dc_DN_c_T_cd = sum(bsxfun(@times, T_cd, prod3(DN_c, T_cd)),1);

        Dgamma_d = Dsqd_noise_d ...
                    + Dn_d ...
                    - 2 * T_dc_DN_cd ...
                    + T_dc_DN_c_T_cd;
        DB_c = prod3(A_cd,tr(DN_cd));          
        A_cd_Dgamma_d = dprod3(A_cd, Dgamma_d);
        A_cd_Dgamma_d_A_dc = prod3(A_cd_Dgamma_d, A_cd');
        T_cd_inv_gamma_d = bsxfun(@times, T_cd, inv_gamma_d);
        C_c_h_c = C_c * h_c;

        % comment means checked for noise and w_0, conditioned on D
        % terms being correct. Note that DN_cd was a zeroes stack for
        % this case, need to check for derivs wrt w_c.
        glogL = 0.5*(...
            - sum(bsxfun(@times, inv_gamma_d, Dgamma_d),2) ... 
            + trace3(A_cd_Dgamma_d_A_dc) ... 
            - 2*trace3(prod3(T_cd_inv_gamma_d, tr(DN_cd))) ... 
            + trace3(prod3(T_cd_inv_gamma_d * T_cd', DN_c)) ... %
            + 2*trace3(prod3(C_c, DB_c)) ...
            - trace3(prod3(C_c * C_c', DN_c)) ... %
            + 2*prod3(prod3(g_c', DN_cd), h_d) ... %
            - prod3(prod3(g_c', DN_c), g_c) ... %
            + prod3(dprod3(h_d', Dgamma_d), h_d) ... %
            - 2*prod3(prod3(h_c', DB_c), g_c) ... %
            - 2*prod3(prod3(C_c_h_c', DN_cd), h_d) ... % 
            + 2*prod3(prod3(C_c_h_c', DN_c), g_c) ...
            - 2*prod3(prod3(h_c', A_cd_Dgamma_d), h_d) ... %
            + 2*prod3(prod3(h_c', DB_c), C_c_h_c) ... %
            - prod3(prod3(C_c_h_c', DN_c), C_c_h_c) ... %
            + prod3(prod3(h_c', A_cd_Dgamma_d_A_dc), h_c) ... %
            );


        glogL = reshape(glogL,num_derivs,1); 
    else
        glogL = [];
    end
    
    
    for name_ind = 1:length(names)
        name = names{name_ind};
        temp_hypersamples(sample_ind).(name) = eval(name);
    end

end
 
gp.hypersamples = temp_hypersamples;

function Dn_d = Dn_d_fn(grad_hp_inds, num_data, num_derivs, ...
    n_d, lambda_ind, w0_inds)


Dn_d = zeros(1, num_data, num_derivs);

for pre_ind = 1 : num_derivs
    
    ind = grad_hp_inds(pre_ind);
    
    is_output_scale = ismember(ind, lambda_ind);
    if is_output_scale
        
        % we actually want the derivative wrt log(output_scale)

        Dn_d(:, :, pre_ind) = 2 * n_d;
        
        continue;
    end
    
    is_w0 = ismember(ind, w0_inds);
    if is_w0
       
        % we actually want the derivative wrt log(w_0)
        
        Dn_d(:, :, pre_ind) = -0.5 * n_d;
        
        continue;
    end
end

function DN_c = DN_c_fn(grad_hp_inds, ...
    num_c, num_dims, num_derivs, ...
    N_c, tw_c, ...
    w_stack_c0, dist_stack_c, sqd_dist_stack_c, ...
    X_c_inds, w_c_inds)
% DN_c(:, :, i) is the derivative of N_c wrt the hyperparameter
% corresponding to grad_hp_inds(i). 



DN_c = zeros(num_c, num_c, num_derivs);

for pre_ind = 1 : num_derivs
    
    ind = grad_hp_inds(pre_ind);
        
    [is_X_c_ind, X_c_ind] = ismember(ind, X_c_inds);
    if is_X_c_ind
        
        [c_ind, dim_ind] = ind2sub([num_c, num_dims], X_c_ind);

        DN_c_vec = N_c(:, c_ind) ...
            .* w_stack_c0(:,c_ind,dim_ind).^(-1)...
            .* dist_stack_c(:,c_ind,dim_ind);
        
        DN_c(:, c_ind, pre_ind) = DN_c_vec;
        DN_c(c_ind, :, pre_ind) = DN_c_vec';
        
        continue;
    end
    
    [is_w_c_ind, w_c_ind] = ismember(ind, w_c_inds);
    if is_w_c_ind
        
        [c_ind, dim_ind] = ind2sub([num_c, num_dims], w_c_ind);
        
        inv_ws = w_stack_c0(:,c_ind,dim_ind).^(-1);
        
        % we actually want the derivative wrt log(w_c - 0.5*w_0)

        DN_c_vec = tw_c(c_ind,dim_ind) * 0.5 * N_c(:, c_ind) ...
            .* inv_ws .* (sqd_dist_stack_c(:,c_ind,dim_ind) .* inv_ws - 1) ;
        
        DN_c(:, c_ind, pre_ind) = DN_c_vec;
        DN_c(c_ind, :, pre_ind) = DN_c_vec';
        DN_c(c_ind, c_ind, pre_ind) = 2 * DN_c(c_ind, c_ind, pre_ind);
        
        continue;
    end
            
end



function DN_cd = DN_cd_fn(grad_hp_inds, ...
    num_c, num_data, num_dims, num_derivs, ...
    N_cd, w_0, tw_c, ...
    w_stack_cd, dist_stack_cd, sqd_dist_stack_cd, ...
    lambda_ind, w0_inds, X_c_inds, w_c_inds)

DN_cd = zeros(num_c, num_data, num_derivs);

for pre_ind = 1 : num_derivs
    
    ind = grad_hp_inds(pre_ind);
     
    is_output_scale = ismember(ind, lambda_ind);
    if is_output_scale
        
        % we actually want the derivative wrt log(output_scale)

        DN_cd(:, :, pre_ind) = N_cd;
        
        continue;
    end
    
    [is_w0, w0_ind] = ismember(ind, w0_inds);
    if is_w0
        dim_ind = w0_ind;
       
        inv_ws = w_stack_cd(:, :, dim_ind).^(-1);
       
        % we actually want the derivative wrt log(w_0)
       
        DN_cd(:, :, pre_ind) = w_0(dim_ind) ...
            *   0.25 * N_cd .* inv_ws...
            .* (sqd_dist_stack_cd(:, :, dim_ind) .* inv_ws...
            	- 1);
       
        continue;
    end
    
    [is_X_c_ind, X_c_ind] = ismember(ind, X_c_inds);
    if is_X_c_ind
        
        [c_ind, dim_ind] = ind2sub([num_c, num_dims], X_c_ind);

        DN_cd_vec = N_cd(c_ind, :) ...
            .* w_stack_cd(c_ind, :, dim_ind).^(-1)...
            .* -dist_stack_cd(c_ind, :, dim_ind);
        
        DN_cd(c_ind, :, pre_ind) = DN_cd_vec;
        
        continue;
    end
    
    [is_w_c_ind, w_c_ind] = ismember(ind, w_c_inds);
    if is_w_c_ind
        
        [c_ind, dim_ind] = ind2sub([num_c, num_dims], w_c_ind);
        
        % row vec
        inv_ws = w_stack_cd(c_ind, :, dim_ind).^(-1);
        
        % we actually want the derivative wrt log(w_c - 0.5*w_0)

        DN_cd_vec = tw_c(c_ind,dim_ind) * 0.5 * N_cd(c_ind, :) .* inv_ws...
            .* (sqd_dist_stack_cd(c_ind, :, dim_ind) .* inv_ws...
            	- 1) ;
        
        DN_cd(c_ind, :, pre_ind) = DN_cd_vec;
        
        continue;
    end
            
end

function [N_c, N_cd] = improve_X_c_problems(allowed_error, N_c, N_cd)
                           
% these two quantities alone should sort out everything else; no need to
% correct dist_stack etc.

problem_vec = ...
    improve_covariance_conditioning(N_c, [], allowed_error, ...
    'identify_problems');

diag_N_c = diag(N_c);
N_c(problem_vec, :) = 0;
N_c(:, problem_vec) = 0;
N_c(diag_inds(N_c)) = diag_N_c;

N_cd(problem_vec, :) = 0;




