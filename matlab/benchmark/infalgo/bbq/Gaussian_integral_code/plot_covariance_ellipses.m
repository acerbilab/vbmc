% generate random covariance
n = 2;
R = rand(n) - 0.5;
K = R'*R;


% D = eigs(K,1);
% 
% [V, DD] = eig(K);
% d = flipud(diag(DD));
% 
% C = inv((1/d(1) - 1/d(2)) * diag(diag(V(:,end) * V(:,end)')) + 1/d(2) * eye(2));

clf
figure(1)
hact = error_ellipse(K, 'style','k');
hold on

% find conservative axis-aligned covarince

cvx_begin sdp
variable C(n,n) diagonal
minimize(trace(C))
C >= K
cvx_end

hdiag = error_ellipse(C)
set(hdiag, 'Color', colorbrew(1))

% hdiag = error_ellipse(diag(diag(K)))
% set(hdiag, 'Color', colorbrew(1))
% heig = error_ellipse(C)
% set(heig, 'Color', colorbrew(2))

set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');

set(gca, 'XTick',[],'YTick',[])

axis equal
legend([hact, hdiag],'actual cov', 'diag cov')
legend boxoff