function fileList = getAllFilesWithExtension(dirName, extension)
    % Get a list of all files and folders in this folder.
    dirData = dir(dirName);      
    % Find the index for directories
    dirIndex = [dirData.isdir];  
    % Get a list of the files
    fileList = {dirData(~dirIndex).name}';
    % Filter for files with the specified extension
    fileList = fileList(endsWith(fileList, extension));
    % Add pathnames to the files
    if ~isempty(fileList)
        fileList = cellfun(@(x) fullfile(dirName, x), fileList, 'UniformOutput', false);
    end
    % Get a list of the subdirectories
    subDirs = {dirData(dirIndex).name};  
    % Find index of subdirectories that are not '.' or '..'
    validIndex = ~ismember(subDirs, {'.', '..'}); 
    % Loop over valid subdirectories
    for iDir = find(validIndex)                 
        nextDir = fullfile(dirName, subDirs{iDir});    % Get the subdirectory path
        % Recursively call getAllFilesWithExtension and append to fileList
        fileList = [fileList; getAllFilesWithExtension(nextDir, extension)];  %#ok<AGROW>
    end
end