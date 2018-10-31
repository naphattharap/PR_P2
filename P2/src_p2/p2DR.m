
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% P2 - RECONEIXEMENT DE PATRONS  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    REDUCCI� DE DIMENSIONALITAT %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%choose the emotion labels we want to classify in the database
% 0:Neutral 
% 1:Angry 
% 2:Bored 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise
emotionsUsed = [0 1 3 4 5 6 7];  

%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
[imagesData shapeData labels stringLabels] = extractData('../CKDB', emotionsUsed);


%%%%%%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%%%%
grayscaleFeatures = extractFeaturesFromData(imagesData,'grayscale');



%%GSCATTER 3 example. Visualize the first three coordiantes of the data.
%%You can remove this after understanding it! :)
gscatter3(grayscaleFeatures(:,1),grayscaleFeatures(:,2),grayscaleFeatures(:,3),stringLabels,7)

% 1.
% 1.1project them in a space of 3 dimensions using PCA.
[dataProjected, meanProjection, vectorsProjection ] = reduceDimensionality(grayscaleFeatures, 'PCA', 3,labels);
% 1.2 Plotting projected data points and plot by gsscatter3

gscatter3(dataProjected(:,1),dataProjected(:,2),dataProjected(:,3),stringLabels,7);
title("3 dimension by PCA")

% 2.
% 2.1 reshape function to make it have dimension 128 × 128 
reshapedMeanProjection = reshape(meanProjection, [128,128]);
% 2.2 using the imagesc or surfing

imagesc(reshapedMeanProjection);
title("By imagesc function")

surf(reshapedMeanProjection);
title("By surf function")
% Why do the vectors of the new basis look like faces?
% Answer:

% If you now project one image in the new space of dimension 3 
% and find that the projection is (1,2,0), what does it mean
% Answer:

% 3. 
% 3.1 Take the features again and now project them 
% in a space of 300 dimensions using PCA.
[dataProjected300, meanProjection300, vectorsProjection300 ] = reduceDimensionality( grayscaleFeatures, 'PCA', 300,labels  );
%gscatter3(dataProjected300(:,1),dataProjected300(:,2),dataProjected300(:,3),stringLabels,7);
%title("Space of 300 dimension using PCA");

% 3.2 After that, project the new vectors of dimension 300 in a space of dimension 3 using LDA. 
[dataProjected3d, meanProjection3d, vectorsProjection3d] = reduceDimensionality(dataProjected300, 'LDA', 3,labels);

% 3.3 Plot with gscatter3
gscatter3(dataProjected3d(:,1),dataProjected3d(:,2),dataProjected3d(:,3),stringLabels,7);
title("Reduce dimension by LDA");

% 4.
% - Reduce the dimensionality of the data of the entire database with PCA to 2 dimensions.
[dataProjected2, meanProjection2, vectorsProjection2 ] = reduceDimensionality( grayscaleFeatures, 'PCA', 2,labels  );
%imagesc(reshape(grayscaleFeatures(1,:),128,128));
% - Reproject them again to the original space
[ dataReprojected2 ] = reprojectData( dataProjected2 , meanProjection2, vectorsProjection2 );
gscatter3(dataReprojected2(:,1),dataReprojected2(:,2),dataReprojected2(:,3),stringLabels,7);
title("Reduce to 2 dim by PCA");
% Now, take one of the reprojected samples and plot it as an image.
reprojectedImage = dataReprojected2(1, :,:);
imagesc(reshape(reprojectedImage, 128,128));
title("Selected first image from 2 dim");

% Repeat the experiment (with the same image) reducing the dimensionality 
% to 2, 5, 10, 50, 100, 300 and 500 dimensions.
dimensions = [2, 5, 10, 50, 100, 300, 500];
% -  Observe how does this image have changed 
% with respect to the original one
lenDimension = length(dimensions);
meanArrs = zeros(lenDimension, 1);
squaredError = zeros(lenDimension,1);

for i = 1:lenDimension
    fprintf("working on dimensions %0f\n",dimensions(i));
    [dataProjectedDim, meanProjectionDim, vectorsProjectionDim ] ...
        = reduceDimensionality(grayscaleFeatures, 'PCA', ...
                                int32(dimensions(i)),labels  );
    [ dataReprojectedDim ] = reprojectData( dataProjectedDim , ...
                            meanProjectionDim, vectorsProjectionDim);
    figure()
    imagesc(reshape(dataReprojectedDim(1,:), 128, 128));
    title(dimensions(i))

    squaredError(i) = immse(dataReprojectedDim,grayscaleFeatures);
    fprintf("result of dimensions %0f error %0f \n",dimensions(i), squaredError);
end

% 5. Calculate the quadratic error between the images of 
% the reprojected dataset (from the previous exercise) 
% and the original images.
plot(dimensions, squaredError);
fprintf('\n The mean-squared error is %0.0f\n', squaredError);

% 6. Divide the data in test and train. 
% Then, with the train set reduce the dimensionality of the dataset 
% to 300 dimensions using PCA and use SVM and Mahalanobis distance 
% to classify the data. You have to take into account that 
% SVM is a binary classifier and you have to classify the 7 emotions.

% Split data into 2 folder by k-fold cross validation
% it shuffles data and return indexes
K = 2;
totalNumberOfImages = length(imagesData);
indexes = crossvalind('Kfold', totalNumberOfImages, K);
% From the index obtained above, split to train and test
trainImages = imagesData(indexes~=1,:,:);
trainLabels = labels(indexes~=1);

testImages = imagesData(indexes==1,:,:);
testLabels = labels(indexes==1);

fprintf("Confirm train labels %f \n",unique(trainLabels));
fprintf("Confirm test labels %f \n",unique(testLabels));

% Reduce dimension to N dim
% Note!! we can test 300 with mahalanobis due to covariance 
% then we set dim to number of columns that less than number of rows.
classifierDim = 10;
originalImageDim = 128*128;
% Reshape train/test data from 3 dim to 2 dim
reshapedTrainImages = reshape(trainImages,size(trainImages,1),originalImageDim);
[trainDataProjected, meanProjectionArr, vectorsProjectionArr ] ...
        = reduceDimensionality(reshapedTrainImages, 'PCA', classifierDim ,labels  );
    
reshapedTestImages = reshape(testImages,size(trainImages,1),originalImageDim);
[testDataProjected, meanProjectionArr, vectorsProjectionArr ] ...
        = reduceDimensionality(reshapedTestImages, 'PCA', classifierDim ,labels  );
    
% Fit train data to SVM by fitcecoc for multi-classification
svmModel = fitcecoc(trainDataProjected,trainLabels);
% Check in-sample error
isLoss = resubLoss(svmModel);
fprintf('In-sample error %f \n', isLoss);

% Predict test data
predictLabels = predict(svmModel, testDataProjected);

% Calculate accuracy (correct prediction/number of test data)
sumAccuracySvm = 0;
numberOfTestSample = length(predictLabels);
for i=1:numberOfTestSample
   fprintf('Predict %f Actual %f\n', predictLabels(i), testLabels(i));
   if predictLabels(i) == trainLabels(i)
       sumAccuracySvm = sumAccuracySvm + 1;
   end
end

svmAccuracyPercent = (sumAccuracySvm/numberOfTestSample)*100;
fprintf('SVM Accuracy %f percent.\n', svmAccuracyPercent);

% Emotion's index and meaning
% 0:Neutral 
% 1:Angry 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise

% Find indexes for each emotion in train data.
idxTrainEmo0 = find(trainLabels==0);
idxTrainEmo1 = find(trainLabels==1);
idxTrainEmo3 = find(trainLabels==3);
idxTrainEmo4 = find(trainLabels==4);
idxTrainEmo5 = find(trainLabels==5);
idxTrainEmo6 = find(trainLabels==6);
idxTrainEmo7 = find(trainLabels==7);

% Split train data for each emotion
trainEmo0 = trainDataProjected(idxTrainEmo0,:);
trainEmo1 = trainDataProjected(idxTrainEmo1,:);
trainEmo3 = trainDataProjected(idxTrainEmo3,:);
trainEmo4 = trainDataProjected(idxTrainEmo4,:);
trainEmo5 = trainDataProjected(idxTrainEmo5,:);
trainEmo6 = trainDataProjected(idxTrainEmo6,:);
trainEmo7 = trainDataProjected(idxTrainEmo7,:);

% Find distance of test image with group of data for each emotion.
mahaPredictedLabels = zeros(numberOfTestSample,1);
for i=1:numberOfTestSample
    % Get each image from test data.
    currentTestImage = testDataProjected(i,:);
    % Get distance from each emotion group.
    distance0 = mahal(currentTestImage, trainEmo0);
    distance1 = mahal(currentTestImage, trainEmo1);
    distance3 = mahal(currentTestImage, trainEmo3);
    distance4 = mahal(currentTestImage, trainEmo4);
    distance5 = mahal(currentTestImage, trainEmo5);
    distance6 = mahal(currentTestImage, trainEmo6);
    distance7 = mahal(currentTestImage, trainEmo7);
    
    % Find minimum distance
    [v, idx] = min([distance0 distance1 distance3 distance4 ...
                    distance5 distance6 distance7]);
    % Convert index (1, 2, ..., 7) to label of corresponding emotion.
    
    if idx == 1
        %Neutral
        mahaPredictedLabels(i) = 0;
    elseif idx == 2
        %Angry
        mahaPredictedLabels(i) = 1;
    else
        % Other indexes and labels defined in emotion array are the same.
        mahaPredictedLabels(i) = idx;
    end
end


sumAccuracyMahalanobis = 0;
for i=1:numberOfTestSample
   fprintf('Predict %f Actual %f\n', mahaPredictedLabels(i), testLabels(i));
   if mahaPredictedLabels(i) == testLabels(i)
       sumAccuracyMahalanobis = sumAccuracyMahalanobis + 1;
   end
end

fprintf('Mahalanobis Accuracy %f percent.\n', (sumAccuracyMahalanobis/numberOfTestSamples)*100);

