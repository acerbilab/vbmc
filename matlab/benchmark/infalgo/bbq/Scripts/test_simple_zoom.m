fn = @cam;
x_0 = [1, 1];

[minimum, minimum_location, X_data, f_data, gp] = ...
  min_zoom(fn, x_0);
