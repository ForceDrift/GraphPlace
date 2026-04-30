import torch
import torch.nn as nn
from typing import List, Any, Optional

class MacroSorter:
    """
    A wrapper class that loads a PyTorch model (.pt file) to sort macros 
    for placement.
    """
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def _load_model(self, path: str) -> Any:
        try:
            model = torch.load(path, map_location=self.device)
            if isinstance(model, dict) and 'state_dict' in model:
                print(f"Loaded state_dict from {path}. Ensure architecture is set.")
                return model
            
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            return None

    def sort(self, macros: List[Any], features: Optional[torch.Tensor] = None) -> List[Any]:
        """
        Sorts the provided macros using the model's predictions.

        Args:
            macros: List of macro objects to be sorted.
            features: Optional tensor of features for the macros if required by the model.

        Returns:
            Sorted list of macros.
        """
        if isinstance(self.model, dict):
            if 'macro_names' in self.model:
                name_to_rank = {name: i for i, name in enumerate(self.model['macro_names'])}
                return sorted(macros, key=lambda m: name_to_rank.get(getattr(m, 'name', ''), 999999))

        if self.model is None or not hasattr(self.model, '__call__'):
            return sorted(macros, key=lambda m: getattr(m, 'width', 0) * getattr(m, 'height', 0), reverse=True)

        with torch.no_grad():
            if features is not None:
                features = features.to(self.device)
                outputs = self.model(features)
                
                if len(outputs.shape) == 1 or (len(outputs.shape) == 2 and outputs.shape[0] == 1):
                    scores = outputs.squeeze()
                    indices = torch.argsort(scores, descending=True)
                    return [macros[i] for i in indices.cpu().numpy()]
            
        return sorted(macros, key=lambda m: getattr(m, 'width', 0) * getattr(m, 'height', 0), reverse=True)

def get_sorted_macros(model_path: str, macros: List[Any]) -> List[Any]:
    sorter = MacroSorter(model_path)
    return sorter.sort(macros)
