options_defined = true;

for i = 1:numel(required_options)
  if (~exist(required_options{i}, 'var'))
    fprintf(['please define ' required_options{i} '.\n']);
    options_defined = false;
  end
end