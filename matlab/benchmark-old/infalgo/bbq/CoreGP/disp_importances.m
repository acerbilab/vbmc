function disp_importances(importances)

if size(importances,1) == 1
    importances = importances';
end

[dummy, indices] = sort(importances, 1, 'descend');

cellfun(@(name, value) fprintf('\t Input %g\t%g\n', name, value), ...
    num2cell(indices), ...
    arrayfun(@(x) {x}, importances(indices)));