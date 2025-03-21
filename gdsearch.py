from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from preprocessing import actions, X_train, y_train

#model architecture
def create_model(learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(Input(shape=(30, 1662)))
    model.add(Bidirectional(GRU(128, return_sequences=False, activation='relu', kernel_regularizer=L2(0.005))))
    model.add(LayerNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_regularizer=L2(0.005)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


model = KerasClassifier(model=create_model, verbose=0)

#range of hyperparameters to search
param_grid = {
    'model__learning_rate': [0.001, 0.0001, 0.00001],
    'model__dropout_rate': [0.3, 0.4, 0.5],
    'batch_size': [32, 64],
    'epochs': [200, 300]
}


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)


print(f"Best Accuracy: {grid_result.best_score_ * 100:.2f}%")
print(f"Best Parameters: {grid_result.best_params_}")
