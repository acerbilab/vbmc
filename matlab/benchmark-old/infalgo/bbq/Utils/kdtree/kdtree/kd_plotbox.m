function kd_plotbox(node_number,plot_str)

% by pramod vemulapalli 02/07/2010
% inspired by the code from Jan Nunnink, 2003
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
% none_number --- the node that needs to be plotted 
% plot_str    --- the type of plot that one would want 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global tree_cell;


hold on;
if (strcmp(plot_str,'node'))
    % in this case just plot the location of the feature vector 
    plot(tree_cell(node_number).nodevector(1),tree_cell(node_number).nodevector(2),'ro')
else
    % in this case draw the cube enclosing the volume 
    a=tree_cell(node_number).hyperrect(1,:);
    b=tree_cell(node_number).hyperrect(2,:);
    c=[a(1) a(1) b(1) b(1);a(1) b(1) b(1) a(1)];
    d=[a(2) b(2) b(2) a(2);b(2) b(2) a(2) a(2)];
    if tree_cell(node_number).type =='leaf'
        line(c,d, 'color', 'k');
    else
        line(c,d, 'color', 'b');
        if(~isempty(tree_cell(node_number).splitval))
            if(tree_cell(node_number).splitdim==1)
                plot([tree_cell(node_number).splitval tree_cell(node_number).splitval],tree_cell(node_number).hyperrect(:,2),'r-')
            end
            if(tree_cell(node_number).splitdim==2)
                plot(tree_cell(node_number).hyperrect(:,1),[tree_cell(node_number).splitval tree_cell(node_number).splitval],'r-')
            end
        end
    end
end
