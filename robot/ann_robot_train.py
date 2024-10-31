import os
import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
    def __init__(self, 
                 input_dims: int = 9,
                 hidden_layers: List[int] = [32, 16],
                 learning_rate: float = 0.01,
                 model_path: str = 'ann_model.keras'):
        """
        Initialize the Neural Network trainer
        """
        self.input_dims = input_dims
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.model = None
        self.history = None
        
    def scale(self, 
              data: Union[pd.DataFrame, np.ndarray], 
              from_interval: Tuple[float, float], 
              to_interval: Tuple[float, float]=(0, 1)) -> np.ndarray:
        """
        Scale data from one interval to another
        """
        from_min, from_max = from_interval
        to_min, to_max = to_interval
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        denominator = (from_max - from_min + epsilon)
        
        return to_min + (data - from_min) * (to_max - to_min) / denominator
    
    def prepare_data(self, 
                    filepath: str, 
                    input_cols: int = 9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training
        """
        try:
            # Read data
            df = pd.read_csv(filepath)
            
            # Scale different columns according to their ranges
            df.iloc[:, 0:8] = self.scale(df.iloc[:, 0:8], (0, 100))
            df.iloc[:, 8] = self.scale(df.iloc[:, 8], (-180, 180))
            df.iloc[:, 9:] = self.scale(df.iloc[:, 9:], (-5, 5))
            
            # Split into input and output
            x = df.iloc[:, :input_cols]
            y = df.iloc[:, input_cols:]
            
            return x, y
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise
    
    def build_model(self) -> Sequential:
        """
        Build the neural network architecture
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], 
                       activation='relu',  # Changed to ReLU for better performance
                       input_shape=(self.input_dims,)))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            
        # Output layer
        model.add(Dense(2, activation='sigmoid'))
        
        # Use Adam optimizer for better performance
        model.compile(optimizer='adam', 
                     loss='mean_squared_error',
                     metrics=['mae'])  # Added metrics for monitoring
        
        return model
    
    def train(self, 
              x: pd.DataFrame, 
              y: pd.DataFrame, 
              batch_size: int = 1000,
              epochs: int = 1000,
              validation_split: float = 0.2) -> None:
        """
        Train the model with early stopping and checkpoints
        """
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Build or load model
        self.model = self.build_model()
        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)
            print("Loaded existing model weights")
        
        # Train model
        self.history = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True
        )
        
    def plot_training_history(self) -> None:
        """
        Plot training and validation loss
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.plot(self.history.history['mae'], label='MAE')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    try:
        # Initialize trainer
        trainer = NeuralNetworkTrainer()
        
        # Prepare data
        x, y = trainer.prepare_data('move_history.csv')
        
        # Train model
        trainer.train(x, y)
        
        # Plot results
        trainer.plot_training_history()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
