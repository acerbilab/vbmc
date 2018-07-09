function [cV, cR, inv_tr_cR_y, inv_cV_y] = ...
    jitter_correction(jitters, K_st, V, R, y, inv_tr_R_y, inv_V_y)

% [R, inv_tr_R_y, inv_V_y] = ...
%     jitter_correction(jitters, K_st, R, inv_tr_R_y, inv_V_y)
%
% This function corrects the jitter added to the diagonal of the covariance
% matrix of a GP so as to best perform prediction for particular
% predictants. 
%
% OUTPUTS
% - cV: the corrected matrix V.
% - cR: the cholesky factor of cV.
% - (optional) inv_tr_cR_y: inv(cR') * y.
% - (optional) inv_cV_y: inv(cV) * y.
%
% INPUTS
% - jitters: the vector of jitters added to the diagonal of the covariance
%       matrix K.
% - K_st: the covariance between the predictants and the data y. Can be
%       alternatively interpreted as a row vector indicating the importance
%       of data.
% - V: the covariance matrix plus jitter, V = K + diag(jitters)
% - R: the cholesky factor of V.
% - (optional) y: the data.
% - (optional) inv_tr_R_y: inv(R') * y.
% - (optional) inv_V_y: inv(V) * y.

% the larger the column sum of K_st, the more important the relevant datum
% is to performing prediction for the predictants.
[highest_correlation, closest_ind] = max(sum(abs(K_st),1));

cV = V;
cV(closest_ind,closest_ind) = V(closest_ind,closest_ind) ...
                                    - jitters(closest_ind);

% try removing the appropriate jitter
[cR,error_msg] = revisechol(cV,R,closest_ind);

% In this case, we don't muck around with the jitter, it's
% going to have to stay in.
no_can_do = error_msg ~= 0;

if no_can_do
    cV = V;
    cR = R;
end

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

if nargin > 4
    if ~no_can_do
        inv_tr_cR_y = revisedatahalf(R,y,inv_tr_R_y,closest_ind);
    else
        inv_tr_cR_y = inv_tr_R_y;
    end
    
    if nargin > 5
        if ~no_can_do
            inv_cV_y = linsolve(R, inv_tr_R_y, uppr);
        else
            inv_cV_y = inv_V_y;
        end
    end
end