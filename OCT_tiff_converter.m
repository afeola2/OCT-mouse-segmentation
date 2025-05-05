clear, clc; 

dirname=uigetdir; 
fileList = getAllFilesWithExtension(dirname, '.OCT');

% Create TIFFS folder if it does not exist
tiffsFolder = fullfile(dirname, 'TIFFS');
if ~exist(tiffsFolder, 'dir')
    mkdir(tiffsFolder);
end
%% Simply load OCT and save them as TIFFS. No averaging. 

for i=1:length(fileList)
    [im, header] = extractOctData(fileList{i});
    [~, fileName, fileExt] = fileparts(fileList{i});
    
    for j=1:size(im,3)
        imwrite(im(:,:,j), [dirname,'\','TIFFS\' , num2str(i),'_', num2str(j) ,'_',  fileName , '.tiff']); 
        [dirname,'\','TIFFS\' , num2str(i),'_', num2str(j) ,'_',  fileName , '.tiff']
    end 
end 
