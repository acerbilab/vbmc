figure

fh = gcf;

set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), params.width, params.height]); 

set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 
set(gca, 'YGrid', 'off');

x=(-3:0.01:3)';
for i = 1:length(x)
    p(i) = p_fn(x(i));
    q(i) = q_fn(x(i));
    r(i) = r_fn(x(i));
end

hold on
plot(x,q,'r--')
plot(x,r,'b')
axes([-3 3 0 1.2]);
xlabel('$\phi$')

legend('$q(\phi)$','$r(\phi)$')

%matlabfrag('/Documents/SBQ/linearisation_r')
