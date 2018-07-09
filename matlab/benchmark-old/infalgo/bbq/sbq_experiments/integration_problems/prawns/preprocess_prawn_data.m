load sixinputs.mat

direction = directions;
clear directions;



%downsample inputs, coreelation length is ~10 frames.
% Note from David:  Looks like this is supposed to remove the second frame from
% each observation, but actually takes all the observations from a single frame.
for i = 1:numel(theta)
    theta = theta(:, 1:2:end);
    direction = direction(:, 1:2:end);
end

theta = theta{1};
direction = direction{1};

save 'sixinputs_downsampled'
