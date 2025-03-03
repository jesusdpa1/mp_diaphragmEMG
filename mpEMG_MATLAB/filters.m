% Moving window function
function windows = moving_window(tsx, window_size, step)
    windows = {};
    for i = 1:step:length(tsx) - window_size + 1
        windows{end+1} = tsx(i:i + window_size - 1);
    end
end

% Rolling mean function
function ma = rolling_mean(tsx, window_size)
    kernel = ones(window_size, 1) / window_size;
    ma = conv(tsx, kernel, 'valid');
    ma = [ma; zeros(window_size/2, 1)];
    ma = [zeros(window_size/2, 1); ma];
end

% Rolling RMS function
function rms = rolling_rms(tsx, window_size)
    kernel = ones(window_size, 1) / window_size;
    squared = tsx.^2;
    rms = sqrt(conv(squared, kernel, 'valid'));
    rms = [rms; zeros(window_size/2, 1)];
    rms = [zeros(window_size/2, 1); rms];
end

% Butterworth lowpass filter function
function [b, a] = butter_lowpass(f0, fs, order)
    nyq = 0.5 * fs;
    normal_cutoff = f0 / nyq;
    [b, a] = butter(order, normal_cutoff, 'low');
end

% Butterworth lowpass filter application function
function tsx_filtered = butter_lowpass_filter(tsx, f0, fs, order)
    [b, a] = butter_lowpass(f0, fs, order);
    tsx_filtered = filtfilt(b, a, tsx);
end

% Butterworth bandpass filter function
function [b, a] = butter_bandpass(lowcut, highcut, fs, order)
    nyq = 0.5 * fs;
    low = lowcut / nyq;
    high = highcut / nyq;
    [b, a] = butter(order, [low high], 'band');
end

% Butterworth bandpass filter application function
function tsx_filtered = butter_bandpass_filter(tsx, lowcut, highcut, fs, order)
    [b, a] = butter_bandpass(lowcut, highcut, fs, order);
    tsx_filtered = filtfilt(b, a, tsx);
end

% Notch filter function
function [b, a] = notch_filter(f0, Q, fs)
    [b, a] = iirnotch(f0, Q, fs);
end

% Notch filter application function
function tsx_filtered = notch_filter_application(tsx, f0, Q, fs)
    [b, a] = notch_filter(f0, Q, fs);
    tsx_filtered = filtfilt(b, a, tsx);
end

% FFC filter function
function tsx_filtered = ffc_filter(tsx, alpha, fc, fs)
    fs_ffc = round(fs / fc);  % delay expressed in number of samples
    tsx_filtered = tsx + alpha * circshift(tsx, -fs_ffc);  % apply the FFC filter
end