clear all;
close all;

config.subjectID = 'C002';
config.session = 'Session_09';

%% FOLDER ACCESS
config.rootFolder = cd;
dataFolder = [config.rootFolder '\_DATA\' config.subjectID '\' config.session '\'];

% load data
run([dataFolder 'sync.m'])

load([dataFolder '\matlab_' config.subjectID '_AISole_' config.session '_5h.mat']);

files = dir(fullfile(dataFolder, 'activity_camera_*.csv'));
cam_tbl = readtable(fullfile(files(1).folder, files(1).name));
% cam_tbl = readtable(fullfile(files(1).folder, files(1).name));
% cam_tbl.ObservationTime = datetime(cam_tbl.ObservationTime, 'ConvertFrom', 'excel');
% cam_tbl.ObservationTime = timeofday(obsTime);

%%

% Insole
PDShoeL.mAcc = ((PDShoeL.r_ax.^2 + PDShoeL.r_ay.^2 + PDShoeL.r_az.^2).^.5);
PDShoeL.timeSync  = PDShoeL.t - PDShoeL.t(1); 

PDShoeR.mAcc = ((PDShoeR.r_ax.^2 + PDShoeR.r_ay.^2 + PDShoeR.r_az.^2).^.5);
PDShoeR.timeSync  = PDShoeR.t - PDShoeR.t(1);

% Camera
camData = expandCamTable_rev01(cam_tbl,start_rec_time);


% Plot
figure(1)
a = subplot(3,1,1);
plot(PDShoeL.timeSync,PDShoeL.mAcc,'LineWidth',1);
ylim([-inf 10]);
title('Insoles Left');

b = subplot(3,1,2);
plot(camData.timeSync,camData.EventType,'LineWidth',1);
title('Camera Event');
xlim([0 inf]);

c = subplot(3,1,3);
plot(PDShoeR.timeSync,PDShoeR.mAcc,'LineWidth',1);
ylim([-inf 10]);
title('Insoles Right');

figure()
pressureL = max([PDShoeL.p_Arch, PDShoeL.p_Hallux, PDShoeL.p_HeelL, PDShoeL.p_HeelR, PDShoeL.p_Met1, PDShoeL.p_Met3, PDShoeL.p_Met5, PDShoeL.p_Toes], [], 2);
pressureR = max([PDShoeR.p_Arch, PDShoeR.p_Hallux, PDShoeR.p_HeelL, PDShoeR.p_HeelR, PDShoeR.p_Met1, PDShoeR.p_Met3, PDShoeR.p_Met5, PDShoeR.p_Toes], [], 2);
plot(PDShoeR.timeSync,(pressureR./max(pressureR))*10);
hold on
plot(PDShoeL.timeSync,(pressureL./max(pressureL))*10);
hold on
plot(camData.timeSync,camData.EventType,'LineWidth',1);

figure()
plot(PDShoeL.timeSync,PDShoeL.mAcc,'LineWidth',1);
hold on
plot(PDShoeR.timeSync,PDShoeR.mAcc,'LineWidth',1);
hold on
plot(camData.timeSync,camData.EventType,'LineWidth',1);



% figure()
% pressureL = max([PDShoeL.p_Arch, PDShoeL.p_Hallux, PDShoeL.p_HeelL, PDShoeL.p_HeelR, PDShoeL.p_Met1, PDShoeL.p_Met3, PDShoeL.p_Met5, PDShoeL.p_Toes], [], 2);
% pressureR = max([PDShoeR.p_Arch, PDShoeR.p_Hallux, PDShoeR.p_HeelL, PDShoeR.p_HeelR, PDShoeR.p_Met1, PDShoeR.p_Met3, PDShoeR.p_Met5, PDShoeR.p_Toes], [], 2);
% plot(PDShoeR.t,pressureR./max(pressureR));
% hold on
% plot(PDShoeL.t,pressureL./max(pressureL));
% hold on
% plot(camData.timeSync,camData.EventType,'LineWidth',1);
% 
% figure()
% plot(PDShoeL.t,PDShoeL.mAcc,'LineWidth',1);
% hold on
% plot(PDShoeR.t,PDShoeR.mAcc,'LineWidth',1);
% hold on
% plot(camData.timeSync,camData.EventType,'LineWidth',1);



%% Save
syncData = ['sync_' config.subjectID '_AISole_' config.session '.mat'];
fullpath = fullfile(dataFolder, syncData);
save(fullpath, "PDShoeL", "PDShoeR", "camData");


error('STOP')



%% CHECK CAMERA ALIGNMENT

acc = sqrt(PDShoeL.ax.^2 + PDShoeL.ay.^2 + PDShoeL.az.^2);
% find tapping

fs = 270;  % Sampling frequency (Hz) – adjust if needed
min_peak_dist_sec = 0.5;  % Min time between taps
min_peak_dist_samples = round(min_peak_dist_sec * fs);

% Step 1: Find acceleration peaks
[accPeaks, accLocs] = findpeaks(acc, ...
    'MinPeakHeight', 8, ...  % or manually tune
    'MinPeakDistance', min_peak_dist_samples);

accTimes = PDShoeL.timeSync(accLocs);


% Step 2: Find camera tap labels
camTapLocs = find(camData.EventType == 1);
% Reduce to rising edges (in case multiple 1s per tap)
camTimes = camData.timeSync(camTapLocs);


% Step 3: Group taps in sets of 3 (each hour)
n_taps_per_group = 3;
n_groups = floor(min(length(camTimes), length(accTimes)) / n_taps_per_group);

accGroups = reshape(accTimes(1:n_groups*n_taps_per_group), n_taps_per_group, [])';
camGroups = reshape(camTimes(1:n_groups*n_taps_per_group), n_taps_per_group, [])';


time_diff = abs(accGroups - camGroups);  % (acc - cam) → positive = camera late
%drift_per_group = mean(time_diff, 2);
drift_per_group = time_diff(:,1);


% Step 5: Plot results
figure;
plot(1:n_groups, drift_per_group, '-o');
xlabel('Tapping (Hour)');
ylabel('Drift (s)');
title('Camera Time Drift');
grid on;





drift = abs(accTimes(1:length(camTimes)) - camTimes(1:length(camTimes)));

figure;
plot(accTimes(1:3:length(accTimes))/3600, drift_per_group, '-o', 'LineWidth', 1.5);
xlabel('Time (h)');
ylabel('Drift (s)');
title('Camera Time Drift vs. Insole Time');
grid on;

p = polyfit(camTimes, accTimes, 1);

correctedCamTime = polyval([0.9997,0], camData.timeSync);


figure()
plot(PDShoeL.timeSync,PDShoeL.mAcc,'LineWidth',1);
hold on
plot(correctedCamTime,camData.EventType,'LineWidth',1);
