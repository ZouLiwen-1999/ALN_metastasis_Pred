function [x,feats] = Preprocessing_and_FeatureExtraction_E(I,BW)
%% ͨ��ת��
if numel(size(I))>2
    I1 = rgb2gray(I);
else
    I1 = I;
end
I2 = uint8(I1);
% figure;
% subplot 121; imshow(I1); title('Original Image');
% subplot 122; imshow(I2); title('Filtered Image');
%% ��״����������ȡ
[x,feats] = Custom_feats_US(I2,BW);
end

