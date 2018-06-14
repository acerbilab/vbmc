prob_bqr_incr_samples
clf
figure(2)
width = 9;
height = 3.8;

fh = gcf;

set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), width, height]); 

set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 
set(gca, 'YGrid', 'off');



x=(-3:0.01:3)';
num_x = length(x);

p = nan(num_x,1);
q = nan(num_x,1);
r = nan(num_x,1);
for i = 1:num_x
    p(i) = p_fn(x(i));
    q(i) = q_fn(x(i));
    r(i) = r_fn(x(i));
end

hold on
plot(x,q,'r--')
plot(x,r,'b')
axis([-3 3 0 1.2]);
xlabel('$\phi$')

leg = legend('$q(\phi)=\pskinny{y\st}{\vz_d,\phi}$','$r(\phi)=\pskinny{\vz_d}{\phi}$',...
    'Location',[0.285 0.8 0.49 0.119])

%set(leg, 'Position', [0.9 -2.6 1.2 0.5])
legend boxoff


set(0, 'defaulttextinterpreter', 'none')

matlabfrag('~/Docs/SBQ/BQR_oned_plot')