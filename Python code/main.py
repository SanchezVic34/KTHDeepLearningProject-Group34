import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import os
from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

from torch import batch_norm, dropout, dropout_



# Reset Keras Session
def reset_keras():
    set_session = tf.compat.v1.keras.backend.set_session
    clear_session = tf.compat.v1.keras.backend.clear_session
    get_session = tf.compat.v1.keras.backend.get_session
    import gc
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass
    removed_variable = gc.collect()
    #print("\n Removed variable :",removed_variable) # if it's done something you should see a number being outputted
    assert removed_variable <= 22

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
key_order = ['pitch', 'step', 'duration']

# Download of dataset
def download_dataset(data_dir = pathlib.Path('data/maestro-v2.0.0')):
    if not data_dir.exists():
        tf.keras.utils.get_file(
        'maestro-v2.0.0-midi.zip',
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        extract=True,
        cache_dir='.', cache_subdir='data',
    )

# generate_filename : generate the file name fiven a directory
def generate_filename(data_dir):
    return glob.glob(str(data_dir/'**/*.mid*'))

# sample_file : sample a file
def sampling_file(filenames, num):
    return filenames[num]
    
# import_file : import the file after downloading the dataset
def import_file(filenames, num = 1, display_instruments = False, extract_notes = False):
    sample_file = sampling_file(filenames, num)
    pm = pretty_midi.PrettyMIDI(sample_file)
    if display_instruments :
        print('Number of instruments:', len(pm.instruments))
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print('Instrument name:', instrument_name)
    if extract_notes :
        instrument = pm.instruments[0]
        for i, note in enumerate(instrument.notes[:10]):
            note_name = pretty_midi.note_number_to_name(note.pitch)
            duration = note.end - note.start
            print(f'{i}: pitch={note.pitch}, note_name={note_name},'
                    f' duration={duration:.4f}')
    return pm #Pretty midi object for sample the MIDI file


# midi_to_notes : transform a midi file into a dataframe panda with notes
# tip :  midi_file = sample_file
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start
    
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

# to test midi_to_notes
# Run :
# raw_notes = midi_to_notes(sample_file)
# raw_notes.head()

# get_note_name : give the name of a given number of note for a name of file
def get_note_name(sample_file, nb_note = 10) :
    raw_notes = midi_to_notes(sample_file)
    #raw_notes.head()
    get_note_names = np.vectorize(pretty_midi.note_number_to_name)
    sample_note_names = get_note_names(raw_notes['pitch'])
    return(sample_note_names[:nb_note])

# plot_piano_roll : visualize the musical piece
# plot the note pitch, start and end across the length of the track (i.e. piano roll).
def plot_piano_roll(notes: pd.DataFrame, title, count: Optional[int] = None, save = True):
    if count:
        title_fig = f'First {count} notes'
    else:
        title_fig = f'Whole track'
    count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="navy", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title_fig)
    if save :
        if not os.path.exists('Results/Result_'+str(title)):
            os.makedirs('Results/Result_'+str(title))
        plt.savefig("Results/Result_"+str(title)+"/piano_roll_"+str(title)+".png")
    plt.show()

# plot the distribution of notes for the pitch, the step and the duration
def plot_distributions(notes: pd.DataFrame, title, drop_percentile=2.5, save = True):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20, color = 'aquamarine')

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21), color = 'aquamarine')

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21), color = 'aquamarine')
    if save :
        title_file = "distrib_estim_"+str(title)+".png"
        if not os.path.exists('Results/Result_'+str(title)):
            os.makedirs('Results/Result_'+str(title))
        plt.savefig("Results/Result_"+str(title)+"/"+str(title_file))
    
    plt.show()


def plot_multiple_distributions(notes1: pd.DataFrame, notes2: pd.DataFrame, title, drop_percentile=2.5, save = True):
    plt.figure(figsize=[15, 5])
    max_step = max(np.percentile(notes1['step'], 100 - drop_percentile), np.percentile(notes2['step'], 100 - drop_percentile))
    max_duration = max(np.percentile(notes1['duration'], 100 - drop_percentile), np.percentile(notes2['duration'], 100 - drop_percentile))
    plt.subplot(1, 3, 1)
    sns.histplot(notes1, x="pitch", bins=20, color = 'aquamarine', label = "input")
    sns.histplot(notes2, x="pitch", bins=20, color = 'r', alpha = 0.4, label = "predicted")
    plt.legend()
    plt.subplot(1, 3, 2)
    sns.histplot(notes1, x="step", bins=np.linspace(0, max_step, 21), color = 'aquamarine', label = "input")
    sns.histplot(notes2, x="step", bins=np.linspace(0, max_step, 21), color = 'r', alpha = 0.4, label = "predicted")
    plt.legend()
    plt.subplot(1, 3, 3)
    sns.histplot(notes1, x="duration", bins=np.linspace(0, max_duration, 21), color = 'aquamarine', label = "input")
    sns.histplot(notes2, x="duration", bins=np.linspace(0, max_duration, 21), color = 'r', alpha = 0.4, label = "predicted")
    plt.legend()
    if save :
        title_file = "distrib_estim_"+str(title)+".png"
        if not os.path.exists('Results/Result_'+str(title)):
            os.makedirs('Results/Result_'+str(title))
        plt.savefig("Results/Result_"+str(title)+"/"+str(title_file))
    
    plt.show()


# notes_to_midi : create a prettyMIDI object
def notes_to_midi(notes: pd.DataFrame, instrument_name: str, velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    return pm

###### CREATION OF DATASET ######
# dataset_generator : generate the dataset
def dataset_generator_maestro(key_order, filenames, seq_length, batch_size, nb_files, random_file = True, display_dataset_info = False, display_sequence_info = False):
    all_notes = []
    list_file_in_dataset = []
    if random_file :
        for _ in range(nb_files):
            rand_file = np.random.randint(len(filenames))
            all_notes.append(midi_to_notes(filenames[rand_file]))
            print("\n music taken as inputs :", filenames[rand_file])
            list_file_in_dataset.append(filenames[rand_file])

    else :
        for f in filenames[:nb_files]:
            notes = midi_to_notes(f)
            all_notes.append(notes)
            list_file_in_dataset.append(f)
        print("\n music taken as inputs :", filenames[:nb_files])
    
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    
    # Creation of an object dataset from the parsed notes
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    if display_dataset_info:
        print(notes_ds.element_spec)

    n_notes = len(all_notes)
    seq_ds = create_sequences(notes_ds, seq_length)
    if display_sequence_info :
        print(seq_ds.element_spec)
        for seq, target in seq_ds.take(1):
            print('sequence shape:', seq.shape)
            print('sequence elements (first 10):', seq[0: 10])
            print()
            print('target:', target)
    train_ds = param_dataset(n_notes, seq_length, seq_ds, batch_size)
    #dataset_dir = pathlib.Path('dataset/seq_length='+str(seq_length)+':')
    return(train_ds, list_file_in_dataset, n_notes)

def dataset_generator_with_file(key_order, filename, seq_length, batch_size, display_dataset_info = False, display_sequence_info = False):
    all_notes = []
    notes = midi_to_notes(filename)
    all_notes.append(notes)
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    
    # Creation of an object dataset from the parsed notes
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    if display_dataset_info:
        print(notes_ds.element_spec)

    n_notes = len(all_notes)
    seq_ds = create_sequences(notes_ds, seq_length)
    if display_sequence_info :
        print(seq_ds.element_spec)
        for seq, target in seq_ds.take(1):
            print('sequence shape:', seq.shape)
            print('sequence elements (first 10):', seq[0: 10])
            print()
            print('target:', target)
    train_ds = param_dataset(n_notes, seq_length, seq_ds, batch_size)
    #dataset_dir = pathlib.Path('dataset/seq_length='+str(seq_length)+':')
    return(train_ds, n_notes)

# create_sequences : create the sequence structure for the training
def create_sequences(dataset: tf.data.Dataset,  seq_length: int, vocab_size = 128, key_order = key_order
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length+1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                                drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)
    # Normalize note pitch
    def scale_pitch(x):
        x = x/[vocab_size,1.0,1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key:labels_dense[i] for i,key in enumerate(key_order)}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def param_dataset(n_notes, seq_length, seq_ds, batch_size):
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    print("\n buffer size is :", buffer_size)
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))
    
    return(train_ds)

# mse_with_positive_pressure : customed loss function 
# For pitch and duration we need a custom mse that encourages the model to output non negative values
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)


def Network_init_test(seq_length, learning_rate, nodes_list, type_RNN = "LSTM", type_optimizer = "Adam"):
    
    """
    input_shape = (seq_length, 3)
    model = tf.Keras.Sequential()

    inputs = tf.keras.Input(input_shape)
    if type_RNN == "LSTM":
        model.add(tf.keras.layers.LSTM(128,return_sequences=True,
               input_shape=input_shape))
        for i in range(nb_of_layers - 1):
            model.add(tf.keras.layers.LSTM(nodes[i+1],return_sequences=True))
    
    elif type_RNN == "GRU":
        model.add(tf.keras.layers.GRU(128,return_sequences=True,
               input_shape=input_shape))
        for i in range(nb_of_layers - 1):
            model.add(tf.keras.layers.GRU(nodes[i+1],return_sequences=True))
    
    elif type_RNN == "Simple":
        model.add(tf.keras.layers.SimpleRNN(128,return_sequences=True,
               input_shape=input_shape))
        for i in range(nb_of_layers - 1):
            model.add(tf.keras.layers.SimpleRNN(nodes[i+1],return_sequences=True))
    
    model.add()

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }
    """
    
    
    input_shape = (seq_length, 3)

    inputs = tf.keras.Input(input_shape)
    x = inputs
    
    x = tf.keras.layers.LSTM(nodes_list[0], return_sequences = True, input_shape=input_shape)(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LSTM(nodes_list[1])(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.LSTM(nodes_list[2])(x)
    #x = tf.keras.layers.Dense(nodes_list[2])(x)
    #x = tf.keras.layers.Dropout(0.3)(x) 


    outputs = {
    'pitch': tf.keras.layers.Dense(128,name='pitch')(x),
    'step': tf.keras.layers.Dense(1,name='step')(x),
    'duration': tf.keras.layers.Dense(1,name='duration')(x),
    }
  

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }
    if type_optimizer == "Adam" : 
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif type_optimizer == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif type_optimizer == "Adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)

    print(model.summary())

    model.compile(loss=loss,loss_weights={'pitch': 0.05,'step': 1.0,'duration':1.0,},optimizer=optimizer)

    return(model)

def Network_init(seq_length, learning_rate, type_RNN = "LSTM", type_optimizer = "Adam", nb_neurons = 128):
    input_shape = (seq_length, 3)

    inputs = tf.keras.Input(input_shape)
    
    if type_RNN == "LSTM":
        x = tf.keras.layers.LSTM(nb_neurons)(inputs)
    elif type_RNN == "GRU": 
        x = tf.keras.layers.GRU(nb_neurons)(inputs)
    elif type_RNN == "RNNSimple": 
        x = tf.keras.layers.SimpleRNN(nb_neurons)(inputs)
    
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }
    if type_optimizer == "Adam" : 
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif type_optimizer == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif type_optimizer == "Adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)

    print(model.summary())

    model.compile(loss=loss,loss_weights={'pitch': 0.01,'step': 1.0,'duration':1.0,},optimizer=optimizer)

    return(model)


def Network_init_multi_layers(seq_length, learning_rate, nodes_list, type_RNN = "LSTM", type_optimizer = "Adam"):
    
    """
    input_shape = (seq_length, 3)
    model = tf.Keras.Sequential()

    inputs = tf.keras.Input(input_shape)
    if type_RNN == "LSTM":
        model.add(tf.keras.layers.LSTM(128,return_sequences=True,
               input_shape=input_shape))
        for i in range(nb_of_layers - 1):
            model.add(tf.keras.layers.LSTM(nodes[i+1],return_sequences=True))
    
    elif type_RNN == "GRU":
        model.add(tf.keras.layers.GRU(128,return_sequences=True,
               input_shape=input_shape))
        for i in range(nb_of_layers - 1):
            model.add(tf.keras.layers.GRU(nodes[i+1],return_sequences=True))
    
    elif type_RNN == "Simple":
        model.add(tf.keras.layers.SimpleRNN(128,return_sequences=True,
               input_shape=input_shape))
        for i in range(nb_of_layers - 1):
            model.add(tf.keras.layers.SimpleRNN(nodes[i+1],return_sequences=True))
    
    model.add()

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }
    """
    #tf.keras.layers.Dropout(0.3)

    nb_of_layers = len(nodes_list)

    input_shape = (seq_length, 3)
    
    inputs = tf.keras.Input(input_shape)
    x = inputs
    if type_RNN == "LSTM":
        x = tf.keras.layers.LSTM(nodes_list[0],return_sequences=True,
               input_shape=input_shape)(x)
        for i in range(1,nb_of_layers-1): 
            x = tf.keras.layers.LSTM(nodes_list[i],return_sequences=True)(x)
        x = tf.keras.layers.LSTM(nodes_list[nb_of_layers-1])(x)
    elif type_RNN == "GRU": 
        x = tf.keras.layers.GRU(nodes_list[0],return_sequences=True,
               input_shape=input_shape)(x)
        for i in range(1,nb_of_layers-1): 
            x = tf.keras.layers.GRU(nodes_list[i],return_sequences=True)(x)
        x = tf.keras.layers.GRU(nodes_list[nb_of_layers-1])(x)
    elif type_RNN == "Simple": 
        x = tf.keras.layers.SimpleRNN(nodes_list[0],return_sequences=True,
               input_shape=input_shape)(x)
        for i in range(1,nb_of_layers-1): 
            x = tf.keras.layers.SimpleRNN(nodes_list[i],return_sequences=True)(x)
        x = tf.keras.layers.SimpleRNN(nodes_list[nb_of_layers-1])(x)

    outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }
    if type_optimizer == "Adam" : 
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif type_optimizer == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif type_optimizer == "Adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)   #construit la structure

    print(model.summary())

    model.compile(loss=loss,loss_weights={'pitch': 0.05,'step': 1.0,'duration':1.0,},optimizer=optimizer)

    return(model)


def train_network(model, train_ds, nb_epochs = 50):

    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),]

    history = model.fit(
        train_ds,
        epochs=nb_epochs,
        callbacks=callbacks,
    )
    return(history)

def plotting_result(history, title_folder, save =  True):
    plt.figure(1)
    plt.plot(history.epoch, history.history['loss'], label='total loss', color='r')
    plt.xlabel("epochs")
    plt.ylabel('loss')
    plt.grid()
    if save :
        title_file = "loss_plot_"+str(title_folder)+".png"
        if not os.path.exists('Results/Result_'+str(title_folder)):
            os.makedirs('Results/Result_'+str(title_folder))
        plt.savefig("Results/Result_"+str(title_folder)+"/"+str(title_file))
    plt.show()
    return history.epoch, history.history['loss']

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> int:

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

def predict_notes(raw_notes, seq_length, title,  vocab_size, instrument_name, temperature, model: tf.keras.Model, export_prediciton = True, num_predictions = 120):

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

    if export_prediciton :
        out_file = "output_music_"+str(title)+".midi"
        out_pm = notes_to_midi(generated_notes, instrument_name=instrument_name)
        if not os.path.exists('Results/Result_'+str(title)):
            os.makedirs('Results/Result_'+str(title))
        out_pm.write('Results/Result_'+str(title)+'/'+str(out_file))
    
    return generated_notes, out_pm
