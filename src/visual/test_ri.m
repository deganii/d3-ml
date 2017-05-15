

learned_dir ='C:\dev\courses\6.874 Computational Systems Biology\Final-Project\figures\predictions\learned\';
truth_dir ='C:\dev\courses\6.874 Computational Systems Biology\Final-Project\figures\predictions\truth\';
files = dir(strcat(learned_dir,'*.jpg'));      

nfiles = length(files);
vals = zeros(nfiles);



count = 1;
for ii=1:nfiles
    currenttruth = strcat(truth_dir,files(ii).name);
    currentlearned = strcat(learned_dir,files(ii).name);
    currenttruth = imread(currenttruth);
    currentlearned = imread(currentlearned);
    [ri,gce,vi] = compare_segmentations(currenttruth, currentlearned);
    rs(ii) = ri;
    gcs(ii) = gce;
    vs(ii) = vi;
end


rsavg = mean(reshape(rs, 4,8));
vsavg = mean(reshape(vs, 4,8));
figure;
plot(rsavg,'LineWidth',1.25)
hold on;
%plot(gcs,'LineWidth',2)
hold on;
plot(vsavg,'LineWidth',1.25);
xlabel('Epochs')
legend('Probabilistic Rand Index','Variation of Information')