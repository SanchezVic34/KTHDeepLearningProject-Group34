
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

%
% Init data
[hot_data, tempo, m_vel] = encode_nmat(nmat);
X = hot_data(:,1:100);
Y = hot_data(:,2:101);

% Init network parameters
K = 108;
d_values = [16 32 64 128];                                        % Change this value to test various hidden state size

% Init Training parameters
seq_length = 20;
%eta = 0.1;
data_sizes = round(exp((4:0.5:9)));                     % Change this value to test various data sizes
%epochs = floor(30000./data_sizes);

%
% Init measurement storage
All_loss_LSTM = cell([length(d_values) length(data_sizes)]);
All_loss_RNN = cell([length(d_values) length(data_sizes)]);
Rec_dist_RNN = cell([length(d_values) length(data_sizes)]);
Rec_dist_LSTM = cell([length(d_values) length(data_sizes)]);
All_param_RNN = cell([length(d_values) length(data_sizes)]);
All_param_LSTM = cell([length(d_values) length(data_sizes)]);
All_hf = cell([length(d_values) length(data_sizes)]);
All_cf = cell([length(d_values) length(data_sizes)]);
All_hf_RNN = cell([length(d_values) length(data_sizes)]);
labels=strings(length(d_values), 1);
colors = [[0.1,0.1,0.8]; [0.1,0.8,0.1]; [0.5,0.1,0.5]; [0.1,0.5,0.5]; [0.5,0.5,0.1]; [0.1,0.1,0.1]];

%% Training for every hidden state size, and data size

% For each size of hidden layer
for n_net = 1:length(d_values)
    d=d_values(n_net);
    epochs=round(sqrt(d)*1e4./data_sizes*4);
    eta=1/d;
    fprintf("hidden layer size: %d \n", d)
    % Init networks
    LSTM = RNNLSTMclass;
    LSTM = LSTM.initialize(K, d);
    RNN = RNNclass;
    RNN = RNN.initialize(K, d);
    
    for n_train = 1:length(data_sizes)
        data_size = data_sizes(n_train);
        fprintf("     data size: %d \n", data_size)
        % Training
        [LSTMstar, Loss_LSTM, cf, hf] = AdamLSTM(LSTM, hot_data(:,1:data_size), seq_length, eta, epochs(n_train), 200, 0);
        [RNNstar, Loss_RNN, hf_RNN] = AdaGrad(RNN, hot_data(:,1:data_size), seq_length, eta, epochs(n_train), 200, 0);

        % Store final loss
        All_loss_RNN{n_net, n_train} = Loss_RNN;
        All_loss_LSTM{n_net, n_train} = Loss_LSTM;
        
        % Reconstruct music
        x0 = zeros(K,1); x0(1)=1;
        hot_seq_LSTM = [x0 LSTMstar.synth_seq(x0, hf, cf, data_size, 0)]; % Deterministic reconstruction
        nmat_LSTM = decode_X(hot_seq_LSTM, tempo, m_vel);
        hot_seq_RNN = [x0 RNNstar.synth_seq(x0, hf_RNN, data_size, 0)];
        nmat_RNN = decode_X(hot_seq_RNN, tempo, m_vel);
        nmat_ref = decode_X([x0 hot_data(:, 1:data_size)], tempo, m_vel);
        
        % Calculate Chamfer distance
        X_RNN = nmat_RNN(:,1:4); X_LSTM = nmat_LSTM(:,1:4); X_ref = nmat_ref(:,1:4);
        weights = [1 0.5 0 5];
        X_RNN = X_RNN.*weights; X_LSTM = X_LSTM.*weights; X_ref = X_ref.*weights;
        ch_RNN = Chamfer(X_RNN, X_ref);
        ch_LSTM = Chamfer(X_LSTM, X_ref);
        ch_ref = Chamfer([0 0 0 0], X_ref);
        
        % Store everything
        All_hf{n_net, n_train} = hf;
        All_cf{n_net, n_train} = cf;
        All_hf_RNN{n_net, n_train} = hf_RNN;
        Rec_dist_RNN{n_net, n_train} = ch_RNN/ch_ref;
        Rec_dist_LSTM{n_net, n_train} = ch_LSTM/ch_ref;
        All_param_RNN{n_net, n_train} = RNNstar;
        All_param_LSTM{n_net, n_train} = LSTMstar;
        
%         % Calculate reconstruction acc
%         acc_RNN = 0;
%         acc_LSTM = 0;
%         for n_note = 1:length(nmat_ref(:, 1))
%             acc_RNN = acc_RNN + sum(ismember(nmat_RNN(:, 1:4),nmat_ref(n_note, 1:4),'rows'));
%             acc_LSTM = acc_LSTM + sum(ismember(nmat_LSTM(:, 1:4),nmat_ref(n_note, 1:4),'rows'));
%         end
%         n_note = length(nmat_ref(:,1));
%         acc_RNN = acc_RNN/n_note; acc_LSTM = acc_LSTM/n_note;
%         
%         % Store reconstruction accuracy
%         Rec_acc_LSTM{n_net, n_train} = acc_LSTM;
%         Rec_acc_RNN{n_net, n_train} = acc_RNN;
    end
end
%

%% Make plots and save data

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
    semilogx(data_sizes, Ch_dist_RNN, 'Color', colors(n_net,:), 'Marker', 'o')
    hold on
    semilogx(data_sizes, Ch_dist_LSTM, 'Color', colors(n_net,:), 'Marker', '+')
    hold on
end
title('Reconstruction distance')
legend(labels, 'Location', 'northwest')
xlabel('data size')
ylabel('rec acc')

% Save data
save('Reconstruction results/rec_sq20.mat', 'All_loss_LSTM', 'All_loss_RNN', 'Rec_dist_RNN', 'Rec_dist_LSTM', 'All_param_RNN', 'All_param_LSTM', 'All_hf', 'All_cf', 'All_hf_RNN', 'd_values', 'data_sizes')














