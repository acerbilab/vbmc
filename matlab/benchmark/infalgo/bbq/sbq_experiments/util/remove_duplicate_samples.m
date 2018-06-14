function [sample_locations, sample_values] = ...
    remove_duplicate_samples(sample_locations, sample_values)

[sample_values, m] = unique(sample_values);
sample_locations = sample_locations(m,:);