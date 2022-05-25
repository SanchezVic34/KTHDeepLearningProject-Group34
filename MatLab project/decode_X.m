function nmat = decode_X(X, tempo, m_vel)

% Optional variables
if ~exist('tempo', 'var')
    tempo = 90;
end
if ~exist('m_vel', 'var')
    m_vel=60;
end

% Init
[K, nchars] = size(X);
nmat = zeros(ceil(nchars),7);
nmat(:, 5) = m_vel*ones(ceil(nchars), 1); % Velocity

% Create an easy to read x
x = (1:K)*X; x = [x 1]; % Add a separator at the end to avoid troubles

% Run through X
t_step = 30/tempo; % the time step of a demi beat
t_start = 0; % start_time (sec)
d_b = 0; % demi_beat we are at
k_note = 1;
for k = 1:nchars
    if x(k)==1 % Then x(k) is a simple separator
        t_start = t_start + t_step;
        d_b = d_b+1;
    elseif x(k)<21 % Then x(k) indicates a duration
        if x(k-1)==1 % This can happen in generated sequence
            t_start = t_start + t_step;
            d_b = d_b+1;
        end
    else % Then x(k) is a note pitch
        nmat(k_note, 4) = x(k); % pitch
        nmat(k_note, 1) = d_b/2; % onset (beat)
        nmat(k_note, 6) = t_start; % onset (sec)
        % x(k+1) is guaranteed to exist if x(k)~=1
        if x(k+1)>20
            if x(k-1)~=x(k) % This might happen in generated sequence
                dur = 1;
                while x(k+dur)==x(k) % This might happen in generated sequence
                    % If a note is written several times in a row, we count
                    % it as a longer note
                    dur = dur+1;
                end
                % Then we know the duration, and that this is a chord
                nmat(k_note, 2) = 0.5*dur; % duration (beat)
                nmat(k_note, 7) = t_step*dur; % duration (sec)
            end
        else
            % Then the next character indicates a duration or a separator
            % so in both cases:
            nmat(k_note, 2) = 0.5*x(k+1); % duration (beat)
            nmat(k_note, 7) = t_step*x(k+1); % duration (sec)
        end
        k_note = k_note+1;
    end
    
    
end

nmat = nmat(1:k_note-1, :);

end