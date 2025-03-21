
from imports import np, Sequential, LSTM, Dense, Input, Bidirectional, GRU, Conv1D, MaxPooling1D, accuracy_score
from preprocessing import actions, X_train, X_test, y_train, y_test

#total actions
num_actions = len(actions)

#model 1: lstm
model1 = Sequential()
model1.add(Input(shape=(30, 1662)))
model1.add(LSTM(64, return_sequences=False, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(num_actions, activation='softmax'))

#compiling the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#training the model
history1 = model1.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)


# Model 2: Bidirectional LSTM Model
model2 = Sequential()
model2.add(Input(shape=(30, 1662)))
model2.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(num_actions, activation='softmax'))

#compiling the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#training the model
history2 = model2.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)


# Model 3: GRU Model
model3 = Sequential()
model3.add(Input(shape=(30, 1662)))
model3.add(GRU(64, return_sequences=False, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(num_actions, activation='softmax'))

#compiling the model
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#training the model
history3 = model3.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)


# Model 4: CNN-LSTM Hybrid Model
model4 = Sequential()
model4.add(Input(shape=(30, 1662)))
model4.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model4.add(MaxPooling1D(pool_size=2))
model4.add(LSTM(64, return_sequences=False, activation='relu'))
model4.add(Dense(32, activation='relu'))
model4.add(Dense(num_actions, activation='softmax'))

#compiling the model
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#training the model
history4 = model4.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)


#function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    yhat = np.argmax(predictions, axis=1)
    ytrue = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(ytrue, yhat)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    

print("Evaluating Model 1:")
evaluate_model(model1, X_test, y_test)

print("Evaluating Model 2:")
evaluate_model(model2, X_test, y_test)

print("Evaluating Model 3:")
evaluate_model(model3, X_test, y_test)

print("Evaluating Model 4:")
evaluate_model(model4, X_test, y_test)
