from imports import plt, sns, np, Sequential, GRU, Dense, Dropout, Input, L2, classification_report, accuracy_score, confusion_matrix, Adam, Bidirectional, LayerNormalization
from preprocessing import actions, X_train, X_test, y_train, y_test

# total actions
num_actions = len(actions)

#best parameters
best_params = {
    'gru_units': 256,
    'dropout_rate': 0.25,
    'learning_rate': 0.00001,
    'l2_regularization': 0.001,
    'dense_units': 128
}

#model architechture
def create_model(gru_units=256, dropout_rate=0.25, learning_rate=0.00001, l2_regularization=0.001, dense_units=128):
    model = Sequential()
    model.add(Input(shape=(30, 1662)))
    model.add(Bidirectional(GRU(gru_units, return_sequences=False, activation='relu', kernel_regularizer=L2(l2_regularization))))
    model.add(LayerNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu', kernel_regularizer=L2(l2_regularization)))
    model.add(Dropout(0.25))
    model.add(Dense(num_actions, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

#initializing the model with best parameters
model = create_model(**best_params)

#training arameters
batch_size = 16 
epochs = 300 

#training the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

#evaluating the model
predictions = model.predict(X_test)
ypred = np.argmax(predictions, axis=1)
ytrue = np.argmax(y_test, axis=1)

accuracy = accuracy_score(ytrue, ypred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(ytrue, ypred, target_names=actions))

print("Confusion Matrix:")
print(confusion_matrix(ytrue, ypred))

#saving final model
model.save('final_model_gru_best_params.keras')

#visualising training & validation accuracy and loss
plt.figure(figsize=(12, 6))

#plt training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

#plt training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

#visualising Confusion Matrix
conf_matrix = confusion_matrix(ytrue, ypred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
