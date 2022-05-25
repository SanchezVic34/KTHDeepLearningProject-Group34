addpath('MIDI_toolbox/miditoolbox');

% Load data
nmat = readmidi('wtcii01a.mid'); % This gives a 'notematrix'

% Init data
[hot_data, tempo, m_vel] = encode_nmat(nmat);
X = hot_data(:,1:100);
Y = hot_data(:,2:101);

%% A few useful commands

% ----------------------- NOTEMATRIX COLUMNS --------------------------
%
%  ONSET     DURATION    MIDI     MIDI    VELOCITY    ONSET   DURATION
% (BEATS)    (BEATS)   channel  PITCH  (or loudness) (SEC)    (SEC)
%
% ---------------------------------------------------------------------

% writemidi(nmat,'test.mid'); % This writes a note matrix into a midi file

% playsound(nmat) % This plays a notematrix

% pianoroll(nmat) % This displays the notes "like de piano sheet"

% plotdist(pcdist1(nmat)) % This plots the pitch distribution

% plotdist(ivdist1(nmat)) % And this plots the 

% Notematrix are encoded into a (108, n) vector, where n is the number of
% demi beats. The 1 to 108 code is: 
% ---------------------------- X ENCODING ----------------------------------
%         1                      2 : 20                 21 : 108
% end of a demi beat      note duration (if>1)      note MIDI pitch
% --------------------------------------------------------------------------

% [X, tempo, m_vel] = encode_nmat(nmat);
% 
% nmat_decoded = decode_X(X, tempo, m_vel);
% 
% pianoroll(nmat)
% figure
% pianoroll(nmat_decoded)

%% Try to generate a sequence
% rng(25);
% 
% K = 108;
% m = 100;
% sig = 0.01;
% 
% RNN = RNNclass;
% RNN = RNN.initialize(K,m,sig);
% 
% [hot_data, tempo, m_vel] = encode_nmat(nmat);
% 
% [RNNstar, Loss] = AdaGrad(RNN, hot_data, 20, 0.1, 20);
% 
% x0 = zeros(K,1);
% h0 = randn(m,1);
% x0(1)=1;
% hot_seq = [x0 RNNstar.synth_seq(x0, h0, 100)];
% 
% nmat_gen = decode_X(hot_seq, tempo, m_vel);
% 
% pianoroll(nmat_gen)
% playsound(nmat_gen) % To hear a masterpiece

%% Try the new LSTM class
% rng(25);
% 
% % Init network
% K = 108;
% d = 32;
% sig = 0.03;
% 
% testRNN = RNNLSTMclass;
% testRNN = testRNN.initialize(K, d, sig);

% Try the forward pass
% h0 = randn(d,1)*0.1; c0 = randn(d,1)*0.1;
% [loss, p, h, a_hat, a, i, f, o, c] = testRNN.forward(X, Y, h0, c0);
% grads = testRNN.backward(X, Y, p, h, h0, a_hat, a, i, f, o, c, c0);

%{
% Try AdaGrad from the get go
[RNNstar, Loss, cf, hf] = AdaGradLSTM(testRNN, hot_data(:,1:1000), 25, 0.1, 50);

x0 = zeros(K,1);
h0 = hf;
c0 = cf;
x0(1)=1;
hot_seq = [x0 RNNstar.synth_seq(x0, h0, c0, 100)];

nmat_gen = decode_X(hot_seq, tempo, m_vel);

pianoroll(nmat_gen)
% playsound(nmat_gen) % To hear a masterpiece
%}

%
% Try Adam from the get go
% [RNNstar, Loss, cf, hf] = AdamLSTM(testRNN, hot_data(:,1:61), 2, 0.1, 1000);
% 
% x0 = zeros(K,1);
% h0 = hf;
% c0 = cf;
% x0(1)=1;
% hot_seq = [x0 RNNstar.synth_seq(x0, h0, c0, 1000)];
% 
% nmat_gen = decode_X(hot_seq, tempo, m_vel);
% 
% hot_seq_temp = [x0 RNNstar.synth_seq_temperature(x0, h0, c0, 100, 1.3)];
% 
% nmat_gen_temp  = decode_X(hot_seq_temp, tempo, m_vel);

% pianoroll(nmat_gen)

%playsound(nmat_gen) % To hear a masterpiece

%% Test gradients
% rng(25);
% K = 20;
% d = 16;
% sig = 0.03;
% datsize=21;
% test_data = randi([1 K], 1, datsize);
% test_data = test_data==(1:K)';
% X = test_data(:, 1:end-1);
% Y = test_data(:, 2:end);
% LSTM = RNNLSTMclass;
% LSTM = LSTM.initialize(K, d, sig);
% h0 = randn(d,1)*0.0001;
% c0 = randn(d,1)*0.001;
% ngrads = ComputeGradsNumLSTM(X, Y, LSTM, 1e-5, h0, c0);
% [loss, p, h, a_hat, a, i, f, o, c] = LSTM.forward(X, Y, h0, c0);
% grads = LSTM.backward(X, Y, p, h, h0, a_hat, a, i, f, o, c, c0);
% for f = fieldnames(grads)'
%     disp(['Relative error for: ' f{1}])
%     disp(std(ngrads.(f{1})-grads.(f{1}))/std(ngrads.(f{1})) + std(ngrads.(f{1})-grads.(f{1}))/std(grads.(f{1})))
% end
% disp("Test for a part of W: ")
% ngradW_a = ngrads.W(1:d, :);
% gradW_a = grads.W(1:d, :);
% disp(std(ngradW_a-gradW_a)/std(ngradW_a) + std(ngradW_a-gradW_a)/std(gradW_a))

%% Get results

rng(25);

% Init networks
K = 108;
d = 32;
sig = 0.03;
LSTM = RNNLSTMclass;
LSTM = LSTM.initialize(K, d, sig);
RNN = RNNclass;
RNN = RNN.initialize(K, d, sig);

% Training
seq_length = 5;
data_size = 50*seq_length +1;
eta = 0.1;
epochs = 150;
[LSTMstar, Loss_LSTM, cf, hf] = AdamLSTM(LSTM, hot_data(:,1:data_size), seq_length, eta, epochs);
[RNNstar, Loss_RNN, hf_RNN] = AdaGrad(RNN, hot_data(:,1:data_size), seq_length, eta, epochs);

% Plot Losses
steps = 1:length(Loss_LSTM);
figure,
plot(steps, Loss_LSTM, steps, Loss_RNN)
legend("LSTM", "RNN")
xlabel("steps")
ylabel("Loss")

% Generate sequence
x0 = zeros(K,1); x0(1)=1;
figure,
hot_seq_LSTM = [x0 LSTMstar.synth_seq(x0, hf, cf, data_size,0)];
nmat_LSTM = decode_X(hot_seq_LSTM, tempo, m_vel);
pianoroll(nmat_LSTM)
title("Pianoroll LSTM")
figure,
hot_seq_RNN = [x0 RNNstar.synth_seq(x0, hf_RNN, data_size,0)];
nmat_RNN = decode_X(hot_seq_RNN, tempo, m_vel);
pianoroll(nmat_RNN)
title("Pianoroll RNN")
figure,
pianoroll(decode_X(hot_data(:,1:data_size), tempo, m_vel));
title("Pianoroll Data")






