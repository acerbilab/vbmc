% pramod vemulapalli 02/08/2010


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Just a simple demo to show the functionality
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
demo_case = [1,2,3,4];

plot_stuff=1;   % 1 if you want to plot the data
% change to 0 if you donot want to plot anything.

if (plot_stuff) close all; end
clc

rand('seed',1)
dimen=2;
X=rand(200,dimen);
point=0.2*ones(1,dimen);

disp('##### Build Tree #####');

tree = kd_buildtree(X,plot_stuff);

for count=1:max(size(demo_case))
    
    switch demo_case(count)

        case 1

            if (plot_stuff); hold on ; end
            if (plot_stuff); plot(point(1),point(2),'g*','MarkerSize',10); end
            disp('##### Closest Point Fast #####');
            [index_vals,vec_vals,node_number] = kd_closestpointfast(tree,point)
            if (plot_stuff); plot(X(index_vals,1),X(index_vals,2),'y*','MarkerSize',10); end
 
        case 2

            if (plot_stuff); hold on ; end
            if (plot_stuff); plot(point(1),point(2),'go'); end
            disp('##### Closest Point Good #####');
            [index_vals,vec_vals,node_number] = kd_closestpointgood(tree,point)
            if (plot_stuff); plot(X(index_vals,1),X(index_vals,2),'m*','MarkerSize',10); end

        case 3
            point=0.6*ones(1,dimen);
            if (plot_stuff); hold on ; end
            if (plot_stuff); plot(point(1),point(2),'g*','MarkerSize',10); end
            disp('##### N Closest Points #####');
            num_of_points=10;
            [index_vals,dist_vals,vec_vals]  = kd_knn(tree,point,num_of_points,plot_stuff)
            if (plot_stuff);
                plot(X(index_vals,1),X(index_vals,2),'g*');
                dist=sqrt(sum((point-X(index_vals(end),1:2)).^2));
                plot(point(1)+dist*cos(0:0.1:2*pi),point(2)+dist*sin(0:0.1:2*pi),'g-','LineWidth',2)
            end


        case 4

            disp('##### Range Query #####');
            point=0.35*ones(1,dimen);
            range=[-0.1*ones(1,dimen); 0.1*ones(1,dimen)];
            [index_vals,dist_vals,vector_vals] = kd_rangequery(tree,point,range)

            %%% plotting stuff
            if (plot_stuff);
                a=point+range(1,:);
                b=point+range(2,:);
                c=[a(1) a(1) b(1) b(1);a(1) b(1) b(1) a(1)];
                d=[a(2) b(2) b(2) a(2);b(2) b(2) a(2) a(2)];
                plot(point(1),point(2),'k*','MarkerSize',10)
                line(c,d, 'color', 'k','LineWidth',2);
                plot(X(index_vals,1),X(index_vals,2),'k*')
            end

    end

end

if (plot_stuff);
    set(gca,'box','on');
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);    
end 
