function Mean_fn = sensors_mean_fn(hps_struct,hp)

mean_inds = hps_struct.Means;
means = hp(mean_inds);
if size(means,2) > size(means,1)
    means=means';
end
means=[means;nan*means];
 
Mean_fn=@(Xs) means(Xs(:,1));
