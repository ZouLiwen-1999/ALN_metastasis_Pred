clc;clear;
RUN_ME_FIRST;
Path = '------'; % Path of original images
Files1=dir(Path);
Feats = [];
Patients = [];
for i=1:length(Files1)
    filename1 = Files1(i).name;
    if strcmp(filename1,'.') || strcmp(filename1,'..')|| contains(filename1, '2')
        continue;
    else
        try
            %% read masks
            BW1 = imread(['------' filename1]); % Path of masks
            if numel(size(BW1)) > 2
                BW1 = imbinarize(rgb2gray(BW1));
            else
                BW1 = imbinarize(BW1); 
            end
            BW1 = ~BW1;
            BW1 = bwmorph(BW1,'clean'); 
            %% read image
            I  = imread([Path filename1]);
            [feat, featn] = Preprocessing_and_FeatureExtraction(I,BW1);
            fprintf(['Extracting shape and texture features from ' filename1 '\n']);
            Feats = [Feats; feat];
            Patients = [Patients; cellstr(filename1(1:strfind(filename1,'1')-1))];
        catch ErrorInfo
            fprintf(['Extracting features failed from ' filename1 '\n']);
        end
        
    end
    
end


[m,p]=size(Feats);
feat_cell=num2cell(Feats,p); 
feat_cell = [Patients feat_cell];
feats = ['ID' featn];
result=[feats;feat_cell];
% Save handcrafted features
s=xlswrite(['------.xlsx'],result);
