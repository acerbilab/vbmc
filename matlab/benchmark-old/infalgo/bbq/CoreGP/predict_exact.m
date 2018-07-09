function mean_out = predict_exact(q, r, prior)
% [mean, sd] = predict(sample, prior, r_gp, opt)
% - q requires fields
% * mean
% * cov
% - r requires fields
% * mean
% * cov
% - prior requires fields
% * means
% * sds

Yot_qr = 0;
yot_r = 0;
for ir = 1:numel(r)
    for iq = 1:numel(q)
        Yot_qr = Yot_qr + ...
                    q(iq).weight * r(ir).weight *...
                    mvnpdf([q(iq).mean; r(ir).mean], ...
                        [prior.means; prior.means], ...
                        repmat(diag(prior.sds.^2), 2)...
                        + blkdiag(q(iq).cov, r(ir).cov));

    end
    
    yot_r = yot_r + ...
        r(ir).weight *...
        mvnpdf(r(ir).mean, prior.means, diag(prior.sds.^2) + r(ir).cov);
end


mean_out = Yot_qr / yot_r;