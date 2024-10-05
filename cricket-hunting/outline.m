% File: mouse_cricket_capture_detector.m

% 1. Setup and Initialization
videoFile = 'example.mp4'; % TODO: Replace with actual video file path
reader = vision.VideoFileReader(videoFile);

% Parameters (can be adjusted based on specific setup)
blobAreaRange = [100, 5000]; % Min and max blob area
catchDuration = 0.5; % seconds
distanceThreshold = 20; % pixels
velocityThreshold = 5; % pixels per frame

% Set up object detector
detector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 50);

% Set up multi-object tracker
tracker = vision.MultiObjectTracker('AssignmentThreshold', 30, 'ConfirmationParameters', [4 5]);

% Initialize variables
timestamps = [];
lastNumObjects = 0;
potentialCatchStart = -1;
lastCricketPosition = [];

% 2. Main Processing Loop
while ~isDone(reader)
    % Read and preprocess frame
    frame = step(reader);
    grayFrame = rgb2gray(frame);
    
    % 3. Object Detection
    foregroundMask = step(detector, grayFrame);
    
    % Clean up the mask
    cleanMask = imopen(foregroundMask, strel('disk', 3));
    cleanMask = imclose(cleanMask, strel('disk', 10));
    
    % Find and filter blobs
    cc = bwconncomp(cleanMask);
    stats = regionprops(cc, 'Area', 'Centroid', 'BoundingBox');
    validBlobs = [stats.Area] > blobAreaRange(1) & [stats.Area] < blobAreaRange(2);
    
    % Get centroids and bounding boxes of valid blobs
    centroids = vertcat(stats(validBlobs).Centroid);
    bboxes = vertcat(stats(validBlobs).BoundingBox);
    
    % 4. Object Tracking
    [tracks, ~] = step(tracker, centroids, bboxes);
    
    % 5. Catch Event Detection
    numObjects = size(tracks, 1);
    
    if numObjects == 2  % Assuming were tracking mouse and cricket
        % Calculate distance between objects
        distance = norm(tracks(1).Centroid - tracks(2).Centroid);
        
        % Calculate cricket velocity (assuming cricket is the smaller object)
        [~, cricketIdx] = min([stats(validBlobs).Area]);
        cricketPosition = tracks(cricketIdx).Centroid;
        if ~isempty(lastCricketPosition)
            cricketVelocity = norm(cricketPosition - lastCricketPosition);
        else
            cricketVelocity = Inf;
        end
        lastCricketPosition = cricketPosition;
        
        % Check for catch conditions
        if distance < distanceThreshold && cricketVelocity < velocityThreshold
            if potentialCatchStart == -1
                potentialCatchStart = reader.CurrentTime;
            elseif reader.CurrentTime - potentialCatchStart >= catchDuration
                % 6. Result Recording
                timestamps = [timestamps; reader.CurrentTime];
                potentialCatchStart = -1;
            end
        else
            potentialCatchStart = -1;
        end
    elseif numObjects < lastNumObjects
        % Alternative detection method based on object disappearance
        if potentialCatchStart == -1
            potentialCatchStart = reader.CurrentTime;
        elseif reader.CurrentTime - potentialCatchStart >= catchDuration
            timestamps = [timestamps; reader.CurrentTime];
            potentialCatchStart = -1;
        end
    else
        potentialCatchStart = -1;
    end
    
    lastNumObjects = numObjects;
    
    % 7. (Optional) Visualization
    % Uncomment the following lines to visualize the tracking
    % annotatedFrame = insertObjectAnnotation(frame, 'rectangle', bboxes, 1:size(bboxes,1));
    % imshow(annotatedFrame);
    % title(['Frame: ' num2str(reader.CurrentTime)]);
    % drawnow;
end

% 8. Output Results
fprintf('Catch events detected at:\n');
fprintf('%f\n', timestamps);

% Optional: Save results to a file
% save('catch_events.mat', 'timestamps');