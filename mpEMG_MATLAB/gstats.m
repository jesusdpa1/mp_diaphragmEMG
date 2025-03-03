function tsx_dc_removed = dc_removal(tsx, method)
    % Remove the DC component from a time series signal.
    %
    % Args:
    %   tsx (vector): Input time series signal.
    %   method (string): Method to use for DC removal. Options: 'mean', 'median'.
    %
    % Returns:
    %   tsx_dc_removed (vector): Time series signal with DC component removed.
    %
    % Raises:
    %   Error: If an invalid method is provided.

    if strcmp(method, 'mean')
        tsx_dc_removed = tsx - mean(tsx);
    elseif strcmp(method, 'median')
        tsx_dc_removed = tsx - median(tsx);
    else
        error('Invalid method. Choose ''mean'' or ''median''.');
    end
end

function normalized_tsx = min_max_normalization(tsx)
    % Normalize tsx using min-max normalization.
    %
    % Args:
    %   tsx (vector): Input tsx to be normalized.
    %
    % Returns:
    %   normalized_tsx (vector): Normalized tsx.

    min_val = min(tsx);
    max_val = max(tsx);
    normalized_tsx = (tsx - min_val) / (max_val - min_val);
end