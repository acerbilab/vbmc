function sd_lengthscales = likelihood_laplace( like_func, laplace_mode, failsafe_sds )
% Estimate posterior uncertainty about the lengthscale hyperparameters,
% using the Laplace approximation on a likelihood function.
%
% David Duvenaud
% February 2012

    % Find the Hessian.
    laplace_sds = Inf;
    try
        laplace_sds = sqrt(-1./hessdiag( like_func, laplace_mode));
    catch e; 
        e;
    end

    % A little sanity checking, since at first the length scale won't be
    % sensible.
    bad_sd_ixs = isnan(laplace_sds) | isinf(laplace_sds) | (abs(imag(laplace_sds)) > 0);
    if any(bad_sd_ixs)
        warning(['Infinite or positive lengthscales, ' ...
                'Setting lengthscale variance to prior variance']);
        laplace_sds(bad_sd_ixs) = failsafe_sds(bad_sd_ixs);
    end


end

end
