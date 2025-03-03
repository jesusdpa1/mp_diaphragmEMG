function le_mean = mean_envelope(tsx, window_size, rectify_method)
    % Calculate the linear envelop using the moving average
    %
    % Args:
    %   tsx (vector): Input signal
    %   window_size (int): Window size
    %   rectify_method (string, optional): Rectification method. Defaults to "power".
    %
    % Returns:
    %   le_mean (vector): LE moving average

    if strcmp(rectify_method, 'power')
        tsx_rectified = tsx.^2;
    elseif strcmp(rectify_method, 'abs')
        tsx_rectified = abs(tsx);
    else
        error('Invalid rectify method. Must be ''power'' or ''abs''.');
    end
    le_mean = rolling_mean(tsx_rectified, window_size);
end

function le_rms = rms_envelope(tsx, window_size, rectify_method)
    % Calculate the linear envelop using the moving RMS
    %
    % Args:
    %   tsx (vector): Input signal
    %   window_size (int): Window size
    %   rectify_method (string, optional): Rectification method. Defaults to "power".
    %
    % Returns:
    %   le_rms (vector): LE moving RMS

    if strcmp(rectify_method, 'power')
        tsx_rectified = tsx.^2;
    elseif strcmp(rectify_method, 'abs')
        tsx_rectified = abs(tsx);
    else
        error('Invalid rectify method. Must be ''power'' or ''abs''.');
    end
    le_rms = rolling_rms(tsx_rectified, window_size);
end

function le_lp = lp_envelope(tsx, fc, fs, rectify_method)
    % Calculate the linear envelop using a Low pass filter
    %
    % Args:
    %   tsx (vector): Input signal
    %   fc (float): Cut-off frequency
    %   rectify_method (string, optional): Rectification method. Defaults to "power".
    %
    % Returns:
    %   le_lp (vector): LE moving RMS

    if strcmp(rectify_method, 'power')
        tsx_rectified = tsx.^2;
    elseif strcmp(rectify_method, 'abs')
        tsx_rectified = abs(tsx);
    else
        error('Invalid rectify method. Must be ''power'' or ''abs''.');
    end
    le_lp = butter_lowpass_filter(tsx_rectified, fc, fs);
end

function le_tdt = tdt_envelope(tsx, fc, fs, order)
    % Extracts the envelope of a time series signal using a series of processing steps.
    %
    % Args:
    %   tsx (vector): Input time series data.
    %   fc (float): Cutoff frequency (Hz).
    %   fs (float): Sampling frequency (Hz).
    %   order (int): Filter order. Minimum: 2. Default: 4.
    %
    % Returns:
    %   le_tdt (vector): Envelope of the time series (same shape as input).

    tsx_rectified = sqrt(tsx.^2);
    tsx_lowpass_filtered = butter_lowpass_filter(tsx_rectified, fc, fs, order);
    sqrt_signal = sqrt(tsx_lowpass_filtered.^2);
    sqrt_signal = sqrt_signal - mean(sqrt_signal);
    le_tdt = sqrt_signal;
end

function distances = chebyshev_distances(X)
    distances = pdist(X, 'chebychev');
end

function sampen = fsampen(tsx, dim, r)
    % Calculate the sample entropy of a time series using Chebyshev distance.
    %
    % Parameters:
    %   tsx (vector): Time series data.
    %   dim (int): Embedding dimension.
    %   r (float): Tolerance value.
    %
    % Returns:
    %   sampen (float): Sample entropy of the time series.

    tsx_matrix = zeros(length(tsx) - dim, dim + 1);
    for i = 1:length(tsx) - dim
        tsx_matrix(i, :) = tsx(i:i + dim);
    end

    matrix_B = squareform(pdist(tsx_matrix(:, 1:dim), 'chebychev'));
    matrix_A = squareform(pdist(tsx_matrix(:, 2:dim + 1), 'chebychev'));

    B = sum(matrix_B(:) <= r);
    A = sum(matrix_A(:) <= r);

    if B == 0
        result = inf;
    else
        result = -log(A / B);
    end

    if isinf(result)
        result = -log(2 / ((length(tsx) - dim - 1) * (length(tsx) - dim)));
    end
    sampen = result;
end