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

load('Reconstruction results/rec_sq20_final.mat')

labels=strings(length(data_sizes), 1);
% for n_net = 1:length(d_values)
%     figure,
%     for n_data = 1:length(data_sizes)
%         plot(All_loss_LSTM{n_net, n_data})
%         hold on
%         labels(n_data) = sprintf('data size=%d', data_sizes(n_data));
%     end
%     d=d_values(n_net);
%     title(sprintf('Training Loss d=%d', d))
%     legend(labels, 'Location', 'northwest')
%     xlabel('step')
%     ylabel('Loss')
%     
% end

%% Use this to plot pianorolls of specific n_net, n_train
[hot_data, tempo, m_vel] = encode_nmat(nmat);

n_net = 4; n_train = 6;
LSTM = All_param_LSTM{n_net,n_train};
RNN = All_param_RNN{n_net, n_train};
data_size = data_sizes(n_train);
h0 = All_hf{n_net,n_train};
c0 = All_cf{n_net,n_train};
h0_RNN = All_hf_RNN{n_net, n_train};
x0 = zeros(108,1); x0(1)=1;

hot_LSTM = [x0 LSTM.synth_seq(x0, h0, c0, data_size, 0)];
hot_RNN = [x0 RNN.synth_seq(x0, h0_RNN, data_size, 0)];
nmat_LSTM = decode_X(hot_LSTM, tempo, m_vel);
nmat_RNN = decode_X(hot_RNN, tempo, m_vel);
nmat_ref = decode_X([x0 hot_data(:, 1:data_size)], tempo, m_vel);
figure,
pianoroll(nmat_ref)
title('Training data')
figure,
pianoroll(nmat_LSTM)
title('LSTM roll')
figure,
pianoroll(nmat_RNN)
title('RNN roll')










