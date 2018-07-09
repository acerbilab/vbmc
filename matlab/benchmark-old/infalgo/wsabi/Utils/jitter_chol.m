function [val] = jitter_chol(covmat)
passed = false;
jitter = 1e-8;
val = 0;
while ~passed
    if (jitter > 100000)
        val = chol(eye(size(covmat)));
        break
    end
        try
            val = chol(covmat + ...
            jitter*eye(size(covmat)));
            passed = true;
        catch ME
            jitter = jitter*1.1;
            passed = false;
        end
end
end