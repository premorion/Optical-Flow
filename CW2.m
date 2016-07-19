function p = CW2(pathname, prefix, first, last, digits, suffix, start)

% If the flows' file is in the same folder as the matlab function, use this
% line
load('flows.mat');

% This line was used to load the flows' file from the server
%load('Z:\student-projects1\ml\2014\comp3085\flows.mat');

% Preload the matrix containing all coloured images
%load('fooMatrix.mat');

% Preload matrix containing all the frame-to-frame distances
%load('distanceMatrix.mat');

% threshold used to minimize edges in the distance matrix graph
threshold = 52;

% Process the images into a matrix
fooMatrix = load_sequence_color(pathname,prefix,first,last,digits,suffix,0.3);

% Obtain dimensions of the image's matrix
[x1,y1,z1,nImages] = size(fooMatrix);

% Calculate the distance Matrix
distanceMatrix = calcDistanceMatrix(fooMatrix);

% Apply a threshold to the distanceMatrix to reduce the number of
% connections between frames
distanceMatrix_threshold = distanceMatrix;
distanceMatrix_threshold(distanceMatrix > threshold) = 0;

% Gabe's face
% Show the default image where the user will introduce mouse input
imshow(fooMatrix(:,:,:,start));
[x,y] = getpts;

% Round these points, as we will have problems obtaining the flow with
% float points
x = round(x);
y = round(y);

% Sparse the matrix
% distanceMatrix_threshold = sparse(distanceMatrix_threshold);

% Obtain the minimun spantree, this will be used to compute a combined tree
% that will be used to find the shortest path between frames
[minMatrix,~] = graphminspantree(sparse(distanceMatrix));

% This function combines the graphs and obtains the lower triangular part
% of the matrix
combinedTree = combineGraphs(minMatrix, distanceMatrix_threshold, distanceMatrix);
combinedTree = tril(combinedTree);

% Uncomment this line to display a biograph showing how the nodes are
% distributed
%view(biograph(combinedTree,[],'ShowArrows','off'));

% During our loop, we are going to modify the initial coordinates of the
% user's input, we inicialice them before.
xStart = x;
yStart = y;

% We create a matrix that will contain all the final points of each flow's
% path.
flowValues = zeros(nImages,2);

% k inicialization
k=0;

% We compute each shortest path from the user's selected node to the other
% nodes in our graph
[~,path,~] = graphshortestpath(sparse(combinedTree),start,'Directed',false);

% Loop inicialization, for each node, we will look to the shortest path
% from the user's image selection to all the nodes in our graph. Once we
% have our path, we will traverse our vector finding the optical flow
% between each node with the following one.
flowPoints=zeros(nImages,40);
for nNodes = 1:nImages
    % We need to obtain the path, as graphshortestpath computes the paths
    % in a cell, we need to obtain it using this type of notation.
    pathtemp = path(1,nNodes,:);
    pathtemp = [pathtemp{1,:}];
    
    % Firstly, we need to inicialice our coordinates every time we compute
    % a new path
    x = xStart;
    y = yStart;
    
    % This loop goes from every value of our path and computes the
    % opticalFlow having into account the opticalFlow created from the
    % lastest pair of nodes.
    for i = 1 : length(pathtemp)-1
        if (pathtemp(i) > pathtemp(i+1))
            k=((pathtemp(i)-1)*(pathtemp(i)-2))/2 + pathtemp(i+1);
            flow = flows_a(:,:,:,k);
            pointFlow = flow(round(y(1)),round(x(1)),:);
            x(1) = x(1) + pointFlow(1);
            y(1) = y(1) + pointFlow(2);
        else
            k=((pathtemp(i+1)-1)*(pathtemp(i+1)-2))/2 + pathtemp(i);
            flow = flows_a(:,:,:,k);
            pointFlow = -flow(round(y(1)),round(x(1)),:);
            x(1) = (x(1) + pointFlow(1));
            y(1) = (y(1) + pointFlow(2));
        end
    end
    % Once we have the final point, we storage it in our flowValues matrix
    flowValues(nNodes,:) = [x(1) y(1)];
end

% Finally, when we have all our points, we look for the pair that is closest
% to the final point
index = dsearchn(flowValues,[x(2) y(2)]);
pathtemp = path(1,index,:);
pathtemp = [pathtemp{1,:}];

% Obtain the images that belong to the path that has the closest point
renderPath = fooMatrix(:,:,:,pathtemp);


% Convert does images into a video
videoName = char('videoPath.avi');
convert2Video(renderPath, videoName);

saveOutput= zeros(x1,y1,length(pathtemp)); 
for i = 1: length(pathtemp)
    saveOutput(:,:,i) = rgb2gray(renderPath(:,:,:,i));
end
save_sequence(saveOutput, 'pathImages\','corrected_images', 0, 4);

%%%%Advanced Section%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%SLOW MOTION

%%%THIS CODE IS COMMENTED BECAUSE IN MY COMPUTER I DID NOT HAVE THE AMOUNT
%%%OF MEMORY NEEDED TO RENDER IT
% k = 1;
% imageStack = zeros(x1,y1,z1,length(pathtemp),length(pathtemp)*10);
% for i = 1 : length(pathtemp)-1
%         if (pathtemp(i) > pathtemp(i+1))
%             k=((pathtemp(i)-1)*(pathtemp(i)-2))/2 + pathtemp(i+1);
%             flow = flows_a(:,:,:,k);
%             pointFlow = flow(round(y(1)),round(x(1)),:);
%             x(1) = x(1) + pointFlow(1);
%             y(1) = y(1) + pointFlow(2);
%         else
%             k=((pathtemp(i+1)-1)*(pathtemp(i+1)-2))/2 + pathtemp(i);
%             flow = flows_a(:,:,:,k);
%             pointFlow = -flow(round(y(1)),round(x(1)),:);
%             x(1) = (x(1) + pointFlow(1));
%             y(1) = (y(1) + pointFlow(2));
%         end
%         for j = 1:2:10
%             im1 = fooMatrix(:,:,:,pathtemp(i));
%             im2 = fooMatrix(:,:,:,pathtemp(i+1));
%             imageSlo = warpFLColor(im1, im2, x(1)+j, y(1)+j);
%             imageStack(:,:,:,i,j) = imageSlo;
%         end
% end
% 
% imageStack2 = zeros(x1,y1,z1,length(pathtemp)*length(pathtemp)*10);
% for indI = 1:size(imageStack(:,:,:,:,:),4)
%     
%     for indJ = 1:size(imageStack(:,:,:,:))
%         imageStack(:,:,:,i*j) = imageStack(:,:,:,i,j);
%     end
% end


% 
% videoName = char('videoSlo.avi');
% convert2Video(imageStack2, videoName);



%%%AUDIO

convert2VideoAudio();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function distMatrix = calcDistanceMatrix(fooMatrix,threshold)

[~,~,~,n] = size(fooMatrix);
distMatrix = zeros(n,n);
for i = 1: n
    for j = 1: i-1
        ecludian = sqrt(sum(sum(sum(((fooMatrix(:,:,:,i) - fooMatrix(:,:,:,j)).^2)))));
        distMatrix(i,j) = ecludian;
        distMatrix(j,i) = ecludian;
    end
end
end


function aux = convert2Video(fooMatrix, videoName)

fprintf('Loading Video...  ');
writerObj = VideoWriter(videoName);
writerObj.FrameRate = 2;
open(writerObj);


for i = 1: (size(fooMatrix,4))

    % Write image into the video object
    writeVideo(writerObj,fooMatrix(:,:,:,i));

end

fprintf('Done!\n');
fprintf('Video saved at %s\n',pwd);
end

function combined = combineGraphs(minMatrix, distanceMatrix_threshold, distanceMatrix)

% As our distanceMatrix_threshold would erase some important paths (it can
% isolate one node or a group of them and create an island) we would try to
% create two masks, the first one will convert all the values greater than 0
% to true, and the same we will do with our distanceMatrix_threshold.

mask1 = minMatrix > 0;
mask2 = distanceMatrix_threshold > 0;

% We need to create a new matrix combining the two calculated and the
% original distanceMatrix, this will fix islands that appear in our
% thresholded matrix.

combined = distanceMatrix .* (mask1 | mask2);


end

function aux = convert2VideoAudio()

fprintf('Loading Video and Audio...  ');
% Load audio file
[audioPacman, ~] = audioread('audio.wav');

% Create a VideoFileReader object to modify and introduce
videoFReader = vision.VideoFileReader('videoPath.avi');
videoFWriter = vision.VideoFileWriter('videoaudio.avi','FrameRate',videoFReader.info.VideoFrameRate,'AudioInputPort', true);

% Just to make sure, we obtain the number of frames in the video reading
% first the first frame, matlab needs this to ensure that, in the case the
% video has a variable frame rate, it obtains the correct number of frames.
Obj = VideoReader('videoPath.avi');
lastFrame = read(Obj, inf);
numFrames = Obj.NumberOfFrames;

for i=1:numFrames
    videoFrame = step(videoFReader);
    step(videoFWriter, videoFrame, audioPacman);
end

release(videoFReader);
release(videoFWriter);

fprintf('Done!\n');
fprintf('Video saved at %s\n',pwd);
end

