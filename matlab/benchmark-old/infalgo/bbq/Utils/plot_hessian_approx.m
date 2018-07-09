function plot_hessian_approx(laplace_V, tl_gp_hypers, samples)


    tl_gp = set_gp('gaussian', 'constant', [], samples.locations, samples.tl,1);

        % Assume the mean is the same as the mode.
            laplace_mode = tl_gp_hypers.log_input_scales;

            % Specify the likelihood function which we'll be taking the hessian of:
            like_func = @(log_in_scale) exp(log_gp_lik2( samples.locations, ...
                samples.tl, ...
                tl_gp, ...
                -12, ...
                log_in_scale, ...
                tl_gp_hypers.log_output_scale, ...
                0));

    % Plot the log-likelihood surface.
    figure(11); clf;
    hrange = linspace(-10, 10, 1000 );
    for t = 1:length(hrange)
        vals(t) = like_func(hrange(t));
    end
    actual=plot(hrange, vals, 'k'); hold on;
    y=get(gca,'ylim');
    h_peak=plot([laplace_mode laplace_mode],y, 'b--');

    % Plot the laplace-approx Gaussian.
    rescale = like_func(laplace_mode)/mvnpdf(0, 0, laplace_V);
    laplace_h=plot(hrange, rescale.*mvnpdf(hrange', laplace_mode, laplace_V), 'b');
    xlabel('log input scale');
    ylabel('likelihood');
    title('Lengthscale laplace approximation');

        set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off', 'FontSize', 10); 
    set(gcf, 'color', 'white'); 
    set(gca, 'YGrid', 'off');
    legend([ actual, laplace_h], {'actual likelihood surface', 'laplace approximation'});
    legend boxoff

end

function log_l = log_gp_lik2(X_data, y_data, gp, log_noise, ...
                             log_in_scale, log_out_scale, mean)
    % Evaluates the log_likelihood of a hyperparameter sample at a certain
    % set of hyperparameters.
    gp.grad_hyperparams = false;
    
    % Todo:  Make a function to package a hyperparameter sample into an array,
    % like the opposite of disp_hyperparams.  Use it to replace this next part,
    % and part of train_gp.m
    sample(gp.meanPos) = mean;
    sample(gp.logNoiseSDPos) = log_noise;
    sample(gp.input_scale_inds) = bound(log_in_scale, -100, 100);
    sample(gp.output_scale_ind) = log_out_scale;
    gp.hypersamples(1).hyperparameters = sample;
    
    gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 1);
    log_l = gp.hypersamples(1).logL;
end
