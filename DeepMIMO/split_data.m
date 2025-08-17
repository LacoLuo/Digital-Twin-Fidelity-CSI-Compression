clear
clc

%% Read the test data indices
M = readtable("DeepMIMO_datasets\Boston5G_3p5_real\test_data_idx.csv");
M = table2array(M);

%% Load the original and estimated channels
load("DeepMIMO_datasets\Boston5G_3p5_real\channel.mat")
load("DeepMIMO_datasets\Boston5G_3p5_real\est_channel.mat")

%% Extract the testing channels
all_channel = all_channel(M, :, :, :);
all_channel_est = channel_est(M, :, :, :);

%% Save the testing data
save("DeepMIMO_datasets\Boston5G_3p5_real\test_channel.mat", "all_channel");
save("DeepMIMO_datasets\Boston5G_3p5_real\test_est_channel.mat", "all_channel_est");