import pydub, random, scipy, numpy, python_speech_features, sys, librosa, matplotlib, pandas, keras
from pydub.playback import play
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB;
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def getAllAudio():
    
    with open("Participants.txt", "r") as myfile:
        participants=myfile.readlines()

    audio_files = []
    for participant in participants:
        participant = participant.replace("\n", "")
        audio_files.append(participant.split("."))

    audio_files_path = "Number Classification Audio/"

    audio_list = []
    for file in audio_files:
        for i in range(5):
            file_name = audio_files_path + str(i+1) + "/" + str(i+1) + "_" + file[0] + "." + file[1]
            try:
                audio_list.append(pydub.effects.normalize(pydub.AudioSegment.from_file(file_name, format=file[1])))
            except:
                audio_list.append(pydub.effects.normalize(pydub.AudioSegment.from_file(file_name, format="m4a")))

    return audio_list

def getMaxDBFS(audio):
    
    interval_time_ms = 20
    max_dbfs = 10000
    timestamp = 0
    
    for i in range(len(audio) // interval_time_ms):

        slice_loudness = audio[i*interval_time_ms: (i+1)*interval_time_ms].dBFS
        
        if abs(slice_loudness) < abs(max_dbfs):
            max_dbfs = audio[i*interval_time_ms: (i+1)*interval_time_ms].dBFS
            timestamp = int((i+0.5)*interval_time_ms)

    return max_dbfs, timestamp

def isolateWord(audio):
    
    max_loudness, timestamp = getMaxDBFS(audio)
    silence_threshold = -50

    chunks = pydub.silence.split_on_silence(audio, min_silence_len=50, silence_thresh=silence_threshold, keep_silence=0)

    for chunk in chunks:
        if chunk.dBFS <= -30 or len(chunk) <= 50:
            chunks.remove(chunk)

    """
    longest_chunk = pydub.AudioSegment.silent(duration=1)
    for chunk in chunks:
        if len(chunk) > len(longest_chunk):
            longest_chunk = chunk
    """

    return chunks

def findMFCC(pydub_audio):
    pudub_audio_array = pydub.AudioSegment.get_array_of_samples(pydub_audio)
    numpy_audio_array = numpy.array(pudub_audio_array)
    numpy_audio_array = numpy.array(numpy_audio_array).astype(numpy.float32)

    target_features = 40
    mfcc_features = librosa.feature.mfcc(y=numpy_audio_array, n_mfcc=target_features)
    mfcc_processed = numpy.mean(mfcc_features.T,axis=0)
    #print(mfcc_features)
    #print(mfcc_processed)
    
    return mfcc_processed

def load_pydub_file(pydub_file_string, file_extention):

    pydub_file = isolateWord(pydub.effects.normalize(pydub.AudioSegment.from_file(pydub_file_string, format=file_extention)))
    print("loading", pydub_file_string)

    return pydub_file[0]

def build_model_graph(num_labels, input_shape=(40,)):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        
    return model

def trainModel(features):
    
    x = numpy.array(features.feature.tolist())
    y = numpy.array(features.class_label.tolist())

    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = \
        train_test_split(x, yy, test_size=0.2, random_state = 127)

    num_labels = yy.shape[1]
    filter_size = 2

    model = build_model_graph(num_labels)

    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]*100

    num_epochs = 500
    num_batch_size = 32
    
    model.fit(x_train, y_train, batch_size=num_batch_size, \
        epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(score[1]))

    print("Save Model? (Y/N)")
    response = input()
    if response == "Y":
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
    else:
        print("Not saving model")

    return model

def getSpectralCentroids(audio_array):

    sc_array = []

    for file in audio_array:
        #Convert from pydub file to numpy array
        pudub_audio_array = pydub.AudioSegment.get_array_of_samples(file)
        numpy_audio_array = numpy.array(pudub_audio_array)
        numpy_audio_array = numpy.array(numpy_audio_array).astype(numpy.float32)

        #Find and print spectral centroid
        sc = librosa.feature.spectral_centroid(numpy_audio_array)
        sc_array.append(sc)

def create_single_feature(numpy_array):

    array = []
    array.append(numpy_array)
    
    return numpy.array(array)

def toResult(result_array):
    res = numpy.where(result_array[0] == max(result_array[0]))
    print(result_array)
    return (res[0][0] + 1)

def main():
    master_files = getAllAudio()

    master_word_files = []
    for i in range(len(master_files)):
        master_word_files.append(isolateWord(master_files[i]))
        getSpectralCentroids(master_word_files[i])

    #Find MFCCs
    features = []
    for i in range(len(master_word_files)):
        for j in range(len(master_word_files[i])):
            features.append([findMFCC(master_word_files[i][j]), str(i+1)])

    featuresDB = pandas.DataFrame(features, columns=["feature", "class_label"])

    print(featuresDB)

    model = Sequential()

    doTrainModel = True
    
    if doTrainModel: model = trainModel(featuresDB)
    else:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")
        
    done = False

    while not done:

        try:
            print("Enter the file to be analyzed (exit+0 to close):")
            file_name, ext = input().split("+")
            
            if file_name == "exit":
                done = True
            else:
                print("My best guess is that it was a", str(toResult(model.predict(create_single_feature(findMFCC(load_pydub_file(file_name, ext)))))))

        except Exception as err:
            print("SOMETHING WENT WRONG. HERE'S THE ERROR MESSAGE:")
            print(str(err))

if __name__ == "__main__":
    main()
