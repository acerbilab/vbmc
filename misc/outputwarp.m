function y = outputwarp(y,optimState,options)
%OUTPUTWARP Apply output warping to training outputs.

if isfield(optimState,'warp_thresh') && ~isempty(optimState.warp_thresh)
    thresh = optimState.warp_thresh;
    idx = y < thresh;
    y(idx) = thresh - sqrt(thresh - y(idx));        
end