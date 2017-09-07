T = readtable('data.txt');
A = table2array(T);
B = A(2:end,2:end);
B = B';
TB = B(1:46,:);
Clear = B(47:end);
Y1 = ones(46,1);
Y2 = zeros(31,1);
Y = cat(1,Y1,Y2);
R = cellfun(@str2num,B);
%lamda = [0.13717*2:-0.00002:0];

%a1 = ones(6859,1);
%a2 = zeros(6859,1);
%lamda = [a1 a2];
Gene_names = A(2:end,1);

Top_10_pure_storer = cell(10,1);
Top_10_weighted_storer = cell(10,1);
Total_weights = cell(10,1);

for icount = [1:1:10]
    
    Top_10_pure_storer{icount} = zeros(13718,1);
    Top_10_weighted_storer{icount} = zeros(13718,1);
    Total_weights{icount} = zeros(13718,1);

    lamda = [0.13717*icount:-0.00001*icount:0];
    [x,info] = Adlas(R,Y,lamda);
    [I1 I2] = sort(x);
    Ordered_genes = Gene_names(I2);
    
    Top_10_genes = I2(end-9:end);
    Top_10_pure_storer{icount}(Top_10_genes) = Top_10_pure_storer{icount}(Top_10_genes) + 1;
    
    Top_10_weighted_storer{icount}(Top_10_genes) = Top_10_weighted_storer{icount}(Top_10_genes) + [1:1:10]';
    
    Total_weights{icount} = abs(Total_weights{icount}) + x;
    
end


%indexes = find(x > 0);
%significant_transcripts_all = headings(indexes);
%significant_transcripts_all = string(significant_transcripts_all)';
%non_zero_weightings = x(indexes)';
%master_all = [significant_transcripts_all; non_zero_weightings];
%master_indexes = [indexes; non_zero_weightings'];

%index = find(x > 0.0001);
%significant_transcripts_thresholded = headings(index);
%significant_transcripts_thresholded = string(significant_transcripts_thresholded)';
%non_zero_weightings_thresholded = x(index)';
%master_thresholded = [significant_transcripts_thresholded; non_zero_weightings_thresholded];

%[Y,I] = sort(master_thresholded(2,:));
%B = master_thresholded(:,I);

save('TB_clear_OWL_lambda_05');

fileID = fopen('TB_clear_OWL_all_lambda_05','w');
fprintf(fileID,'%6s %12s\r\n','Transcript','Weightings');
fprintf(fileID,'%6s %12s\r\n',master_all);
fclose(fileID);

fileID = fopen('TB_clear_OWL_all_indexes_lambda_05','w');
fprintf(fileID,'%6s %12s\r\n','Index','Weightings');
fprintf(fileID,'%6s %12s\r\n',master_indexes);
fclose(fileID);

fileID = fopen('TB_clear_OWL_thresholded_lambda_05','w');
fprintf(fileID,'%6s %12s\r\n','Transcript','Weightings');
fprintf(fileID,'%6s %12s\r\n',master_thresholded);
fclose(fileID);

C = B(2,:);
D = zeros(1,46);
for v = 1:46
    D(v) = C(v);
end
