%% extractOctData.m
%   Editable function file for extracting the contents of a .oct file.
%
% Revision history
%   2010.10.26  Created file.
%   2014.12.05  Modified extractOctFunction.m to save variables to memory
%               rather than to files. Created extractOctData.m, I. Campbell
%
% Examples
%   [im, header] = extractOctFunction(filepath)
%
% Contact information
%   Brad Bower
%   bbower@bioptigen.coma
%   Bioptigen, Inc. Confidential
%   Copyright 2010
%
% Edited by
%   Ian Campbell
%   iancampbell@gatech.edu
%   December 2014

%% Function Definition
function [im, header] = extractOctData(filepath)
% Initialize values
scans   = 0; % does not exist in older file versions
frames  = 0; % does not exist in older file versions
dopplerFlag = 0;    % does not exist in older file versions

%% Extract OCT data for current file
fid             = fopen(filepath);


%% Read file header
magicNumber         = fread(fid,2,'uint16=>uint16');
magicNumber         = dec2hex(magicNumber);
header.magicNumber = magicNumber;

versionNumber       = fread(fid,1,'uint16=>uint16');
versionNumber       = dec2hex(versionNumber);
header.versionNumber = versionNumber;

keyLength           = fread(fid,1,'uint32');
key                 = char(fread(fid,keyLength,'uint8'));
dataLength          = fread(fid,1,'uint32');
if (~strcmp(key','FRAMEHEADER'))
    errordlg('Error loading frame header','File Load Error');
end

headerFlag          = 0;    % set to 1 when all header keys read
while (~headerFlag)
    keyLength       = fread(fid,1,'uint32');
    key             = char(fread(fid,keyLength,'uint8'));
    dataLength      = fread(fid,1,'uint32');

    % Read header key information
    if (strcmp(key','FRAMECOUNT'))
        header.frameCount      = fread(fid,1,'uint32');
    elseif (strcmp(key','LINECOUNT'))
        header.lineCount       = fread(fid,1,'uint32');
    elseif (strcmp(key','LINELENGTH'))
        header.lineLength      = fread(fid,1,'uint32');
    elseif (strcmp(key','SAMPLEFORMAT'))
        header.sampleFormat    = fread(fid,1,'uint32');
    elseif (strcmp(key','DESCRIPTION'))
        header.description     = char(fread(fid,dataLength,'uint8'));
    elseif (strcmp(key','XMIN'))
        header.xMin            = fread(fid,1,'double'); %mm
    elseif (strcmp(key','XMAX'))
        header.xMax            = fread(fid,1,'double'); %mm
    elseif (strcmp(key','XCAPTION'))
        header.xCaption        = char(fread(fid,dataLength,'uint8'));
    elseif (strcmp(key','YMIN'))
        header.yMin            = fread(fid,1,'double'); %mm
    elseif (strcmp(key','YMAX'))
        header.yMax            = fread(fid,1,'double'); %mm
    elseif (strcmp(key','YCAPTION'))
        header.yCaption        = char(fread(fid,dataLength,'uint8'));
    elseif (strcmp(key','SCANTYPE'))
        header.scanType        = fread(fid,1,'uint32');
    elseif (strcmp(key','SCANDEPTH'))
        header.scanDepth       = fread(fid,1,'double'); %mm
    elseif (strcmp(key','SCANLENGTH'))
        header.scanLength      = fread(fid,1,'double'); %mm
    elseif (strcmp(key','AZSCANLENGTH'))
        header.azScanLength    = fread(fid,1,'double'); %mm
    elseif (strcmp(key','ELSCANLENGTH'))
        header.elScanLength    = fread(fid,1,'double'); %mm
    elseif (strcmp(key','OBJECTDISTANCE'))
        header.objectDistance  = fread(fid,1,'double'); %mm
    elseif (strcmp(key','SCANANGLE'))
        header.scanAngle       = fread(fid,1,'double'); %deg
    elseif (strcmp(key','SCANS'))
        header.scans           = fread(fid,1,'uint32');
    elseif (strcmp(key','FRAMES'))
        header.frames          = fread(fid,1,'uint32');
    elseif (strcmp(key','FRAMESPERVOLUME')) % x104
        header.framesPerVolume = fread(fid,1,'uint32');
    elseif (strcmp(key','DOPPLERFLAG'))
        header.dopplerFlag     = fread(fid,1,'uint32');
    elseif (strcmp(key','CONFIG'))
        header.config          = fread(fid,dataLength,'uint8');
    else
        headerFlag      = 1;
    end         % if/elseif conditional
end             % while loop


%% Correct header info based on scan type
if header.scanType == 6 % mixed mode volume
    errordlg('Mixed Density (''S'') Scans Not Supported.');
    fclose(fid);
    return;
end

fseek(fid,-4,'cof');            % correct for 4-byte keyLength read in frame header loop
fileHeaderLength = ftell(fid);
fseek(fid,0,'bof');
headerBytes     = fread(fid,fileHeaderLength);

%% Read frame data
% Initialize frames in memory, need to modify for mod(lineLength,2)~=0
imageData           = zeros(header.lineLength,header.lineCount,'uint16');
im                  = zeros(cat(2,size(imageData),header.frames*header.scans),'uint16');
% imageData         = zeros(frameCount,lineLength,lineCount,'uint16');
dopplerData         = 0;
imageFrame          = zeros(header.lineLength/2,header.lineCount,'uint16');
if dopplerFlag == 1
    dopplerData     = zeros(header.lineLength,header.lineCount,'uint16');
    dopplerFrame    = zeros(header.lineLength/2,header.lineCount,'uint16');
end

% Insert additional code here for initializing image volume, e.g.
% imageVolume = zeros(frameCount,lineLength,lineCount)

currentFrame        = 1;
frameLines          = zeros(1,header.frameCount);  % for tracking lines/frame in annular scan mode

%     % Generate waitbar
%     hCurrentFileLoad    = waitbar(0,'Loading Current .oct File');
while (currentFrame <= header.frameCount);
    %         if mod(currentFrame,10) == 0
    %             waitbar(currentFrame/frameCount,hCurrentFileLoad);
    %         end     % Only update every other 10 frames
    frameFlag       = 0;        % set to 1 when current frame read

    keyLength       = fread(fid,1,'uint32');
    key             = char(fread(fid,keyLength,'uint8'));
    dataLength      = fread(fid,1,'uint32');

    if (strcmp(key','FRAMEDATA'))
        while (~frameFlag)
            keyLength       = fread(fid,1,'uint32');
            key             = char(fread(fid,keyLength,'uint8'));
            dataLength      = fread(fid,1,'uint32'); % convert other dataLength lines to 'uint32'

            % The following can be modified to have frame values persist
            % Need to modify to convert frameDataTime and frameTimeStamp from byte arrays to real values
            if (strcmp(key','FRAMEDATETIME'))
                header.imheader{currentFrame}.frameDateTime   = fread(fid,dataLength/2,'uint16'); % dataLength/2 because uint16 = 2 bytes
                %                     frameYear       = frameDateTime(1);
                %                     frameMonth      = frameDateTime(2);
                %                     frameDayOfWeek  = frameDateTime(3);
                %                     frameDay        = frameDateTime(4);
                %                     frameHour       = frameDateTime(5);
                %                     frameMinute     = frameDateTime(6);
                %                     frameSecond     = frameDateTime(7);
                %                     frameMillisecond= frameDateTime(8);

            elseif (strcmp(key','FRAMETIMESTAMP'))
                header.imheader{currentFrame}.frameTimeStamp  = fread(fid,1,'double'); % dataLength is 8 for doubles
            elseif (strcmp(key','FRAMELINES'))
                header.imheader{currentFrame}.frameLines(currentFrame)    = fread(fid,1,'uint32');
            elseif (strcmp(key','FRAMESAMPLES'))
                if header.imheader{currentFrame}.frameLines(currentFrame) == 0 % framelines tag not present in some earlier versions of IVVC
                    imageData       = fread(fid,[header.lineLength,header.lineCount],'uint16=>uint16');
                else
                    imageData       = fread(fid,[header.lineLength,header.imheader{currentFrame}.frameLines(currentFrame)],'uint16=>uint16');
                end
                [header.imheader{currentFrame}.imageHeight, header.imheader{currentFrame}.imageWidth] = size(imageData);
            elseif (strcmp(key','DOPPLERSAMPLES'))
                dopplerData     = fread(fid,[header.lineLength,header.frameLines(currentFrame)],'uint16=>uint16');
                [header.imheader{currentFrame}.imageHeight, header.imheader{currentFrame}.imageWidth] = size(dopplerData);
            else
                fseek(fid,-4,'cof');                    % correct for keyLength read
                frameFlag       = 1;
            end % if/elseif for frame information
        end % while (~frameFlag)

        % Image Data
        % These variables can be saved to .mat files or otherwise manipulated in Matlab
        if header.scanType == 2 %Checks if it is a circular scan and trys to pad each image to be the same
            if abs(size(im,2) - size(imageData,2)) == 0
                imageData = flipud(imageData);                                      % Flips images upside down for EDI images
                im(:,:,currentFrame)  = imageData;
            else
                PadLength = abs(size(im,2) - size(imageData,2));                    % Calculates the number of arrays needed to pad
                PadVal = mean(mean(imageData(end-10:end,:)));                       % Calculates the average noise value at the edge of the single to Pad the image with
                imageData = padarray(imageData,[0 PadLength/2],PadVal,'both');      % Pads both sides of the image to keep it centered assumes an even difference
                imageData = flipud(imageData);                                      % Flips images upside down for EDI images
                im(:,:,currentFrame)  = imageData;
            end
        else
            im(:,:,currentFrame)  = imageData;
        end
        % imageFrame(frameIndex,:,:) = imageData;
        if (dopplerFlag == 1)
            im(:,:,currentFrame) = dopplerData;
        end % if to check Doppler flag


        currentFrame    = currentFrame + 1;     % will increase to frameCount + 1
    end % frames while loop
end % volume while loop

%% Shutdown
%     close(hCurrentFileLoad);    % close current file progress bar
%% ADDED SHOULD CROP OFF THE BOTTOM HALF OF THE IMAGE
%im = im(1:1024,:,:); %CHECK


if header.scanType ~= 2
    AdjustSize = 1;       %Value to cut off the size NEEDS TO be multiple of image size typically 2048
    if AdjustSize < 1
        scaleVal = size(im,1)*AdjustSize;
        im = im(1:scaleVal,:,:);
        %Adjust the xMin to accoxunt for cropping half the image.
        header.xMin = header.xMax*AdjustSize;
    else
        scaleVal = size(im,1)*AdjustSize;
        im = im(1:scaleVal,:,:);
    end
end
%% ADDED DENOISE TEST
%     net = denoisingNetwork('DnCNN');
%     %denoisedI = denoiseImage(im1,net);
%     for kk = 1:size(im,3)
%         im(:,:,kk) = denoiseImage(im(:,:,kk),net);
%     end

fclose(fid);

end % end function definition


%A = padarray(imageData,[0 25],0,'both');
