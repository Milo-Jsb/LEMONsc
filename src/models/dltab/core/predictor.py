# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from typing           import Union

# Predictor Class ---------------------------------------------------------------------------------------------------------#
class Predictor:
    """
    ________________________________________________________________________________________________________________________
    Predictor: Handles inference and prediction for deep learning models
    ________________________________________________________________________________________________________________________
    Responsibilities:
    -> Standard prediction for arrays/tensors
    -> Batch prediction using DataLoader
    -> Memory-efficient inference
    ________________________________________________________________________________________________________________________
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, use_amp: bool = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the Predictor.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model   (torch.nn.Module) : Model to use for predictions
        -> device  (torch.device)    : Device to use for inference
        -> use_amp (bool)            : Whether to use Automatic Mixed Precision during inference
        ____________________________________________________________________________________________________________________
        """
        self.model   = model
        self.device  = device
        self.use_amp = use_amp
    
    def predict(self, X: Union[np.ndarray, torch.Tensor, DataLoader]) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Make predictions on new data.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X (array-like or DataLoader) : Input features as array/tensor or DataLoader for batch predictions
        ____________________________________________________________________________________________________________________
        Returns:
        -> predictions (np.ndarray) : Model predictions
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError : If X is None or invalid
        ____________________________________________________________________________________________________________________
        """
        if X is None:
            raise ValueError("X cannot be None")
        
        # Set model to evaluation mode
        self.model.eval()
        
        try:
            # Check if input is a DataLoader for batch prediction
            if isinstance(X, DataLoader):
                return self._predict_dataloader(X)
            
            # Standard prediction for arrays/tensors
            return self._predict_tensor(X)
        
        except (TypeError, ValueError, RuntimeError) as e:
            raise ValueError(f"Error making predictions: {e}") from e
    
    def _predict_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Make predictions on numpy arrays or torch tensors.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X (array-like) : Input features
        ____________________________________________________________________________________________________________________
        Returns:
        -> predictions (np.ndarray) : Model predictions
        ____________________________________________________________________________________________________________________
        """
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        # Move to device
        X = X.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                predictions = self.model(X)
        
        return predictions.detach().cpu().numpy()
    
    def _predict_dataloader(self, dataloader: DataLoader) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Make predictions using a DataLoader for memory-efficient batch processing.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> dataloader (DataLoader) : DataLoader containing batches to predict on
        ____________________________________________________________________________________________________________________
        Returns:
        -> predictions (np.ndarray) : Concatenated predictions from all batches
        ____________________________________________________________________________________________________________________
        """
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Consistent unpacking: expect (X, y) or just X
                if isinstance(batch, (list, tuple)):
                    batch_X = batch[0]
                else:
                    batch_X = batch
                
                batch_X = batch_X.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds = self.model(batch_X)
                    all_predictions.append(preds.detach().cpu())
        
        if not all_predictions:
            raise ValueError("DataLoader returned no batches")
        
        return torch.cat(all_predictions, dim=0).numpy()

#--------------------------------------------------------------------------------------------------------------------------#
