

for i = [1:5]
    
    
    
    
    figure(i);
        clf
    xlab = ['dim_',num2str(i)];
    filename = ['~/Dropbox/papers/sbq-paper/exp_loss_mov_',xlab];

    load(filename, 'mov');
    
    movie(mov(1))
    
    set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off', 'FontSize', 10); 
    set(gcf, 'color', 'w'); 
    set(gca, 'YGrid', 'off');
    
    xlabel(xlab, 'Interpreter', 'latex');
    axis tight
    set(gca, 'YTick',[])
    
    set(gcf, 'units', 'centimeters');
    pos = get(gcf, 'position'); 
    set(gcf, 'position', [pos(1:2), 10, 10]); 
    
    writerObj = VideoWriter([filename,'.avi']);
    writerObj.FrameRate = 2;
    
    open(writerObj);
    
    for j = 1:length(mov)
        writeVideo(writerObj,mov(j));
    end
    close(writerObj);
end
