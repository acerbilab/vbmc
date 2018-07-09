function handle=shaded_sd(X,mean,sd,colour)
%shaded_sd(X,mean,sd,colour)

if nargin<4
    colour=[0.87 0.89 1];
else
    colour=min(1,colour+0.88);
end


%colour=colour+0.5*([1 1 1]-colour);

handle=fill([X;flipud(X)],[mean+sd;flipud(mean-sd)],colour,'EdgeColor',colour);