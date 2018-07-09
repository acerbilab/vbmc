
function print_table( table_title, row_titles, col_titles, data )

[rows, cols] = size(data);

fprintf('%-20s | ', table_title);
for c = 1:cols
    fprintf('%-16s | ', col_titles{c} );
end
fprintf('\n-------------------------------------------------------------------------------------------\n');
for r = 1:rows
    fprintf('%-20s | ', row_titles{r} );
    for c = 1:cols
        fprintf('   %7.3f       | ', data(r,c) );
    end
    fprintf('\n');
end
