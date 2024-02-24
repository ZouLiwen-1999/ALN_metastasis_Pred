% FIRST RUN THIS FILE TO LOAD THE US TOOLBOX

addpath(fullfile(pwd,'config'))
addpath(fullfile(pwd,'C functions'))
addpath(fullfile(pwd,'Classification'),...
        fullfile(pwd,'Classification','classifiers'),...
        fullfile(pwd,'Classification','normalization'),...
        fullfile(pwd,'Classification','ranking'))
addpath(fullfile(pwd,'Data'))
addpath(fullfile(pwd,'Features'),...
        fullfile(pwd,'Features','morphological'),...
        fullfile(pwd,'Features','texture'),...
        fullfile(pwd,'Features','birads'))
addpath(fullfile(pwd,'Preprocessing'),...
        fullfile(pwd,'Preprocessing','contrast'),...
        fullfile(pwd,'Preprocessing','despeckling'),...
        fullfile(pwd,'Preprocessing','domain_transformation'))
addpath(fullfile(pwd,'Segmentation'))
addpath(fullfile(pwd,'Miscellaneous'))
