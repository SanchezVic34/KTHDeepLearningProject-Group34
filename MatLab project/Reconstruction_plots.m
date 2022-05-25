addpath('MIDI_toolbox/miditoolbox');

rng(25);

% Load data
if ~exist('not_loaded', 'var')
    not_loaded=1;
end
if not_loaded
    disp("Loading data")
    nmat = readmidi('MIDI_long.midi'); % This gives a 'notematrix'
    not_loaded=0;
end

load('Reconstruction results/rec_sq50_final.mat')

labels=strings(length(d_values), 1);
colors = [[0.1,0.1,0.8]; [0.1,0.8,0.1]; [0.5,0.1,0.5]; [0.1,0.5,0.5]; [0.5,0.5,0.1]; [0.1,0.1,0.1]];

% Plot minimal loss
figure,
for n_net = 1:length(d_values)
    bloss_RNN = zeros(1, length(data_sizes));
    bloss_LSTM = zeros(1, length(data_sizes));
    for n_data = 1:length(data_sizes)
        bloss_RNN(n_data) = min(All_loss_RNN{n_net, n_data});
        bloss_LSTM(n_data) = min(All_loss_LSTM{n_net, n_data});
    end
    d=d_values(n_net);
    semilogx(data_sizes, bloss_RNN, 'Color', colors(n_net,:), 'Marker', 'o')
    hold on
    semilogx(data_sizes, bloss_LSTM, 'Color', colors(n_net,:), 'Marker', '+')
    hold on
    labels(2*n_net-1)=sprintf('RNN: d=%d', d);
    labels(2*n_net)=sprintf('LSTM: d=%d', d);
end
title('Minimal training loss')
legend(labels, 'Location', 'northwest')
xlabel('data size')
ylabel('min loss')

% Plot reconstruction accuracy
figure,
for n_net=1:length(d_values)
    d = d_values(n_net);
    Ch_dist_RNN = zeros(1,length(data_sizes));
    Ch_dist_LSTM = zeros(1,length(data_sizes));
    for k = 1:length(data_sizes)
        if isempty(Rec_dist_RNN{n_net, k})
            Ch_dist_RNN(k) = 1e6;
        else
            Ch_dist_RNN(k) = Rec_dist_RNN{n_net, k};
        end
        if isempty(Rec_dist_LSTM{n_net, k})
            Ch_dist_LSTM(k) = 1e6;
        else
            Ch_dist_LSTM(k) = Rec_dist_LSTM{n_net, k};
        end
    end
    loglog(data_sizes, Ch_dist_RNN, 'Color', colors(n_net,:), 'Marker', 'o')
    hold on
    loglog(data_sizes, Ch_dist_LSTM, 'Color', colors(n_net,:), 'Marker', '+')
    hold on
end
title('Reconstruction distance')
legend(labels, 'Location', 'northwest')
xlabel('data size')
ylabel('rec acc')


