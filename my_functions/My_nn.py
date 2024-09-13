import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import numpy as np
import visualkeras
import matplotlib.pyplot as plt


class My_nn:
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, model):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        self.model = model
        self.model_name = "modelo"
        self.epochs = 20
        self.batchsize = 16

        self.show_model()

    def show_model(self):
        self.model.summary()
        file_path = f"{self.model_name}"

        plot_model(self.model, f"{file_path}_pm.png", show_shapes=True)
        visualkeras.layered_view(self.model, to_file=f"{file_path}_vk.png", legend=True)

    def train(self):
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batchsize,
            verbose=2,
        )

        return history

    def show_loss_history(self, history):
        # Obtener métricas de entrenamiento y validación
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        train_mae = history.history["mae"]
        val_mae = history.history["val_mae"]
        epochs = range(1, len(train_loss) + 1)

        # Gráfica de Pérdida y Precisión en Conjunto de Entrenamiento vs. Validación
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mae, "b", label="Training MAE")
        plt.plot(epochs, val_mae, "r", label="Validation MAE")
        plt.title("Training and validation MAE")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend()

        plt.show()

    def evaluate(self):
        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        return loss, mae

    def recognize(self):
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def get_report(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_pred, y_true)
        reporte = {
            "Mean Squared Error (MSE)": mse,
            "Mean Absolute Error (MAE)": mae,
            "R-squared (R2)": r2,
        }

        return reporte

    def save_model(self):
        # Guardar los pesos del modelo en un archivo
        self.model.save_weights(f"{self.model_name}_weights.h5")
        # Guardar el modelo completo en un archivo
        self.model.save(f"{self.model_name}.h5")
