import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

NUM_WORDS = 1000
SEQ_LEN = 100
EMBEDDING_SIZE = 100
BATCH_SIZE = 100
EPOCHS = 5
THRESHOLD = 0.5


# generate data from text file
def get_data_from_text():
    df_train = pd.DataFrame(columns=['text', 'sent'])
    df_test = pd.DataFrame(columns=['text', 'sent'])
    text = []
    sent = []

    file = open('smsspamcollection/SMSSpamCollection', 'r', encoding='UTF8')
    lines = file.readlines()
    train_num = int(len(lines) * 0.8)

    for i in range(train_num):
        line = lines[i]
        if line[0] == 's':
            text.append(line[4:].strip())
            sent.append(0)
        else:
            text.append(line[3:].strip())
            sent.append(1)

    df_train['text'] = text
    df_train['sent'] = sent
    text = []
    sent = []

    for i in range(train_num, len(lines)):
        line = lines[i]
        if line[0] == 's':
            text.append(line[4:].strip())
            sent.append(0)
        else:
            text.append(line[3:].strip())
            sent.append(1)

    df_test['text'] = text
    df_test['sent'] = sent

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    return df_train, df_test


if __name__ == '__main__':
    train_df, test_df = get_data_from_text()

    """
    Convert text to numeric data
    """
    # create tokenizer for our data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, oov_token='<UNK>')
    tokenizer.fit_on_texts(train_df['text'])

    # convert text data to numerical indexes
    train_seqs = tokenizer.texts_to_sequences(train_df['text'])
    test_seqs = tokenizer.texts_to_sequences(test_df['text'])

    # pad data up to SEQ_LEN (note that we truncate if there are more than SEQ_LEN tokens)
    train_seqs = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=SEQ_LEN, padding="post")
    test_seqs = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, maxlen=SEQ_LEN, padding="post")

    """
    Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_SIZE),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(EMBEDDING_SIZE, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    """
    Train Model
    """
    log_dir = ".\\logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max')
    callbacks = [tensorboard_callback]
    history = model.fit(train_seqs, train_df['sent'].values, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2,
                        callbacks=callbacks)

    print(model.evaluate(test_seqs, test_df['sent'].values))

    """
    Plot History
    """
    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    """
    Save Model
    """
    model.save('spam_model.h5')
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del model
    del tokenizer