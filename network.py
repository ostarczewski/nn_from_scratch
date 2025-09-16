import numpy as np
from layer import Layer
from loss import Loss
from solver import Solver
from data import Dataset
from typing import List
import copy


class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.loss = None
        self.solver = None
        self.learning_rate = None


    def compile(self, loss: Loss, solver: Solver):
        self.loss = loss
        self.solver = solver


    def forward(self, input: np.ndarray, training=True) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input, training=training)
            # for testing
            # print(input)
        return input


    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input, training=False)
    

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        # computes the derivative of loss wrt y_pred
        output_grad = self.loss.get_grad(y_true, y_pred)
        
        # each layer receives the previous gradient, computes new gradients to pass further 
        # solver updates layers params
        for layer in reversed(self.layers):
            output_grad, param_grad = layer.calculate_gradients(output_grad)
            self.solver.apply_gradients(layer=layer, param_grad=param_grad)


    def train(self, dataset: Dataset, epochs: int, val_dataset: Dataset = None, early_stopping: dict = None, 
              verbose: bool = True):
        # store loss values from training
        history = {
            'loss': [],
            'val_loss': []
        }

        # check if early stopping is active
        early_stopping_active = early_stopping is not None and val_dataset is not None

        # set early stopping parameters
        if early_stopping_active:
            patience = early_stopping.get('patience', 5)  # int
            min_delta = early_stopping.get('min_delta', 0)  # float
            restore_best_parameters = early_stopping.get('restore_best_parameters', True)  # bool

            # val loss tracking for early stopping and best_layers storing
            min_val_loss = float('inf')
            epochs_without_improvement = 0
            best_layers = copy.deepcopy(self.layers)

        batch_num = dataset.get_batch_num()

        # training loop
        for epoch in range(1, epochs+1):
            total_epoch_loss = 0

            for batch_idx, (x_train, y_train) in enumerate(dataset, start=1):
                # forward pass
                y_pred = self.forward(x_train)

                # calculate loss
                batch_loss = np.sum(self.loss.get_loss(y_train, y_pred, vectorized=True))  # sum of all the errors from each observation 
                total_epoch_loss += batch_loss

                # backward pass
                self.backward(y_train, y_pred)

                if verbose:
                    print(f"Epoch {epoch}/{epochs}, batch {batch_idx}/{batch_num} - Batch Loss: {batch_loss/dataset.batch_size:.6f}", end='\r', flush=True)

            avg_epoch_loss = total_epoch_loss / len(dataset)  # works even for smaller last batch

            if verbose:
                print(" "*64, end='\r', flush=True)  # clear the line
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_epoch_loss:.6f}")
            history['loss'].append(avg_epoch_loss)


            # validation data evaluation
            if val_dataset:
                total_val_epoch_loss = 0

                for x_val, y_val in val_dataset:
                    y_pred = self.predict(x_val)
                    total_val_epoch_loss += np.sum(self.loss.get_loss(y_val, y_pred, vectorized=True))
                
                avg_epoch_val_loss = total_val_epoch_loss / len(val_dataset)

                if verbose:
                    print(f"Validation Loss: {avg_epoch_val_loss:.6f}")
                history['val_loss'].append(avg_epoch_val_loss)


                # storing loss values for early stoping and best network parameter tracking
                if early_stopping_active:
                    if avg_epoch_val_loss + min_delta < min_val_loss:
                        # store new best loss value
                        min_val_loss = avg_epoch_val_loss
                        # reset epochs without improvemnt counter
                        epochs_without_improvement = 0
                        # store the best layers
                        best_layers = copy.deepcopy(self.layers)
                    else:
                        epochs_without_improvement += 1


            # early stopping check
            if early_stopping_active:
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} - no validation loss improvemnt for {patience} epochs")

                    # restores best params
                    if restore_best_parameters:
                        self.layers = copy.deepcopy(best_layers)
                        if verbose:
                            print(f"Network parameter values reverted to epoch {epoch-patience}")
                    
                    # exits the training loop
                    break

        # return loss values
        return history



