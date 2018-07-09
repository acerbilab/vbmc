function [X_c, y_c] = spgp_centres(gp, r_gp)
% [X_c, y_c] = spgp_centres(gp, r_gp)
% return estimates of the location and values of the centres for plotting
% purposes

r_y_data = vertcat(gp.hypersamples.logL);
[max_logL, max_ind] = max(r_y_data);

X_c = gp.hypersamples(max_ind).X_c;
opt.sparse = true;
y_c = predict_spgp(X_c, gp, r_gp, opt);
