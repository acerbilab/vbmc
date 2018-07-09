function [m,C,gm,gC] = posterior_spgp(X_star, gp, sample,flag)
% sparse version of posterior_gp
% can construct cells of flags if desired e.g.
% {'var_not_cov','noise_corrected_variance'}

X_data = gp.X_data;

if nargin<3
    sample=1;
end
if nargin<4
    flag = '';
end

% include the extra noise in our predictive variance to reflect the
% fact that we make predicions about the noisy measurements, rather
% than the latent variable
noise_corrected_variance = nargout~=1 && ...
    any(strcmpi(flag,'noise_corrected_variance'));
if noise_corrected_variance
    diag_sqd_noise = gp.diag_sqd_noise;
    % assume noise has zero derivative for now.
end

% actually return a vector of variances in the place of a covariance
% matrix
varnotcov=any(strcmpi(flag,'var_not_cov'));

% If nocov, we do not compute the covariance
nocov = nargout==1 || any(strcmpi(flag,'no_cov'));

% You can work this one out yourself
nomean = any(strcmpi(flag,'no_mean'));

num_dims=size(X_data,2);
num_star=size(X_star,1);

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;


hs = gp.hypersamples(sample).hyperparameters;

Mu = gp.Mu;
DMu = gp.DMu_inputs;




lambda = exp(hs(gp.lambda_ind));
w_0 = exp(hs(gp.w0_inds));
tw_c = exp(gp.hypersamples(sample).log_tw_c);
w_c = bsxfun(@plus, tw_c, 0.5*w_0);
X_c = gp.hypersamples(sample).X_c;
num_c = size(w_c,1);



n_st = lambda^2/(sqrt(prod(2*pi*w_0)));

R_c = gp.hypersamples(sample).R_c;
T_cd = gp.hypersamples(sample).T_cd;

problem_c_inds = all(T_cd' == 0);

inv_gamma_d = gp.hypersamples(sample).inv_gamma_d;
A_cd = gp.hypersamples(sample).A_cd;
h_c = gp.hypersamples(sample).h_c;
y_data_minus_Mu = gp.hypersamples(sample).y_data_minus_Mu;

w_stack_cst = repmat(...
    reshape(w_c,num_c,1,num_dims),...
            1,num_star,1);

dist_stack_cst = ...
    bsxfun(@minus,...
            reshape(X_c,num_c,1,num_dims),...
            reshape(X_star,1,num_star,num_dims));
        
sqd_dist_stack_cst = dist_stack_cst.^2;

N_cst = lambda * 1./(sqrt(prod(2*pi.*w_stack_cst,3))).*...
    exp(-0.5*sum(sqd_dist_stack_cst./w_stack_cst,3));
N_cst(problem_c_inds, :) = 0;

K_std = N_cst' * T_cd;

% this is the K_std of the full GP
%K_std = lambda^2*1/sqrt(2*pi*w_0)*exp(-0.5*bsxfun(@minus, X_star, X_data').^2/w_0);

H_std = bsxfun(@times, K_std, inv_gamma_d);
H_stc = K_std * A_cd';

if ~nomean
    m = Mu(hs, X_star) + H_std * y_data_minus_Mu - H_stc * h_c;     
    % Compute the objective function value at X_star
end

if nargout>1
    
    

    if varnotcov
        
        C = n_st ...
            - sum(K_std .* H_std, 2) ...
            + sum(H_stc.^2, 2);
        
        if noise_corrected_variance
            C = C + diag_sqd_noise(hs, X_star);
        end
        
    elseif ~nocov
        
        tinv_R_c_N_cst = linsolve(R_c, N_cst, lowr);
        K_st = tinv_R_c_N_cst' * tinv_R_c_N_cst;
        K_st(diag_inds(K_st)) = n_st;
        
        C = K_st ...
            - K_std * H_std' ...
            + H_stc * H_stc';
        
        if noise_corrected_variance
            C = C + diag(diag_sqd_noise(hs, X_star));
        end
    end

    
 
    if nargout>2
        
        dist_stack_cst(isinf(dist_stack_cst)) = 1;
        % dN_cst(:, i, j) represents the derivative of N_cst wrt X_st(i, j)
        DN_cst = bsxfun(@times, N_cst, dist_stack_cst./w_stack_cst);
        DK_std = prod3(tr(DN_cst), T_cd);
        
        DH_std = bsxfun(@times, DK_std, inv_gamma_d);
        DH_stc = prod3(DK_std, A_cd');

        % Gradient of the function evaluated at X_star
        %gm = DK(X_star,X_data)*datatwothirds;  
        if ~nomean
            gm = DMu(hs, X_star) + ...
                prod3(DH_std, y_data_minus_Mu) - prod3(DH_stc, h_c);
            
            gm = permute(gm, [1 3 2]);
        end

        if nargout>3
            % The derivative of the diagonal elements of Kst should be zero
            % wrt X_star
            
            
            
            if varnotcov

                gC = - 2 * sum(bsxfun(@times, DK_std, H_std), 2) ...
                    + 2 * sum(bsxfun(@times, DH_stc, H_stc), 2);
                
                gC = permute(gC, [1 3 2]);

            elseif ~nocov
                
                Dtinv_R_c_N_cst = linsolve3(R_c, DN_cst, lowr);
                DK_st = 2 * prod3(tinv_R_c_N_cst', Dtinv_R_c_N_cst);
                DK_st(diag_inds(DK_st)) = 0;

                gC =  DK_st...
                    - 2 * prod3(DK_std, H_std') ...
                    + 2 * prod3(DH_stc, H_stc');

            end
        end
    end
end

if nocov
    C=[];
    gC=[];
end
if nomean
    m=[];
    gm=[];
end

% hack to dodge problems with very small negative values
C = max(eps,C);

