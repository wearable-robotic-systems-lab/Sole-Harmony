function useableTable = expandCamTable(table, rec_time)
% EXPANDCAMTABLE  Expand BORIS interval-based activity annotations
% into a per-sample (100 ms resolution) activity label timeseries, and
% align it to the insole clock.
%
% INPUTS
%   table    - BORIS event table with columns:
%                Behavior          : activity label (string/cell)
%                Duration_s_       : duration of each event (s)
%                ObservationTime   : wall-clock time of each event
%   rec_time - insole recording start time as a UNIX (POSIX) timestamp,
%              read from sync.m
%
% OUTPUT
%   useableTable - table with columns:
%                    Time      : time (s) since the first camera event
%                    EventType : numeric activity code (see switch-case
%                                below for the label -> code mapping)
%                    timeSync  : Time shifted by the camera/insole clock
%                                offset, i.e. on the same clock as the
%                                insole data (PDShoeL/PDShoeR.t)
%
% NOTE: the expansion is done at a fixed rate of 10 samples/second
% (100 ms steps), independent of the original insole sampling rate. Any
% unrecognized/unlabeled event defaults to code -1.

    %% Prepare event durations and total recording duration
    table.Duration_s_ = round(table.Duration_s_, 3);
    durationTot = round(sum(table.Duration_s_, 'omitnan'), 3); % total annotated time (s)

    %% Parse observation (wall-clock) timestamps
    if iscell(table.ObservationTime) || isstring(table.ObservationTime) || ischar(table.ObservationTime)
        obsTime = datetime(table.ObservationTime, 'InputFormat', 'HH:mm:ss');
    else
        obsTime = table.ObservationTime;
    end

    % Time-of-day (duration) for each observation, relative to the first
    start_t = timeofday(obsTime(1));
    tod     = timeofday(obsTime);
    rel_tod = tod - tod(1);
    t_seconds = seconds(rel_tod);

    %% Compute camera-to-insole clock offset
    % Convert the insole recording start (UNIX timestamp) to a datetime,
    % then compare it to the camera's first observation time-of-day on
    % the same calendar day to get the offset between the two clocks.
    dt_insole = datetime(rec_time, 'ConvertFrom', 'posixtime', 'TimeZone', 'America/New_York');
    dt_cam    = dateshift(dt_insole, 'start', 'day') + start_t;
    offset    = seconds(dt_insole - dt_cam); % offset in seconds

    fprintf('Offset: %.2f seconds\n', offset);

    %% Expand each annotated event into 100 ms samples
    rowNum = int64(durationTot * 10); % expected number of samples (10 Hz)
    convertedtempArray = zeros(rowNum + 50000, 2); % preallocate with buffer (time, label)
    i = 1;
    tStart = -1;

    for r = 1:length(table.Behavior)

        duration = table.Duration_s_(r);
        label = lower(table.Behavior{r});

        tempArr = zeros(int64(duration * 10), 2); % (time, label) samples for this event
        for e = 1:length(tempArr(:,1))
            tempArr(e,1) = tStart + 1; % assign time, continuing from previous sample
            tStart = tempArr(e,1);

            % Map activity label to numeric event code.
            % Numeric-string labels ('0'-'4','-1') are passed through
            % unchanged to support annotation files already coded numerically.
            switch label
                case 'sitting'
                    event = 0;
                case 'standing'
                    event = 1;
                case 'walking'
                    event = 2;
                case 'stairs descending'
                    event = 3;
                case 'stairs ascending'
                    event = 4;
                case 'undefined'
                    event = -1;
                case 'other'
                    event = -1;
                case '0'
                    event = 0;
                case '1'
                    event = 1;
                case '2'
                    event = 2;
                case '3'
                    event = 3;
                case '4'
                    event = 4;
                case '-1'
                    event = -1;
                otherwise
                    event = -1;
            end

            tempArr(e,2) = event;
            convertedtempArray(i,:) = tempArr(e,:);
            i = i + 1;
        end
    end

    %% Mark isolated "tapping" events (used as sync markers)
    % A tapping annotation is a single instantaneous event rather than an
    % interval; find the expanded sample closest to its observation time
    % and label it with code 7.
    for r = 1:length(table.ObservationTime)
        label = lower(table.Behavior{r});
        if strcmp(label, 'tapping')
            TimeTap = round(t_seconds(r) * 10);
            if TimeTap >= 0
                [~, TapIDX] = min(abs(convertedtempArray(:,1) - TimeTap));
                convertedtempArray(TapIDX, 2) = 7;
            end
        end
    end

    %% Finalize and align to insole clock
    convertedtempArray = convertedtempArray(1:i-1, :);   % trim unused buffer rows
    convertedtempArray(:,1) = convertedtempArray(:,1) ./ 10; % convert time index back to seconds

    useableTable = array2table(convertedtempArray, 'VariableNames', {'Time','EventType'});
    useableTable.timeSync = useableTable.Time - offset; % align to insole time base

end