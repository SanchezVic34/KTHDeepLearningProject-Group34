function [X, tempo, m_vel] = encode_nmat(nmat)

tempo = mean(nmat(:,2)./nmat(:,7))*60;
m_vel = round(mean(nmat(:,5)));

start = round(nmat(:,1)*2); % Notes start (in demi beat)
dur = round(nmat(:,2)*2); % Notes duration (in demi beat)
note = nmat(:,4); % MIDI note encoding
nchars = 108; % 88 possible notes + 19 digits expressing note duration + 1 separator

X = zeros(nchars, 3*(start(end)+dur(end))); % X will be reduced at the end

prev_start = 0;
curr_t = 1;
for n=1:length(note)
    % First add the needed number separators
    if prev_start~=start(n) % If this is not a chord
        X(1, curr_t:curr_t+start(n)-prev_start-1) = ones(1, start(n)-prev_start); % 1 is the separator
        curr_t = curr_t+start(n)-prev_start;
        prev_start = start(n);
    end
    
    % Then add the note
    X(note(n), curr_t) = 1;
    curr_t = curr_t+1;
    
    % Then add the duration if any
    if dur(n)>1
        X(max(min(dur(n),20),1), curr_t) = 1; % Maximum duration at 20
        curr_t = curr_t+1;
    end
    
end

% Remove the remaining unused space from X
X = X(:, 1:curr_t-1);

end
















