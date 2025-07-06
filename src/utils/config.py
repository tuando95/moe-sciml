import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration management for AME-ODE."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value[k]
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        try:
            return self[key]
        except (KeyError, TypeError):
            return default
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config['training']
    
    @property
    def integration(self) -> Dict[str, Any]:
        """Get integration configuration."""
        return self._config['integration']
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config['data']
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config['evaluation']
    
    @property
    def compute(self) -> Dict[str, Any]:
        """Get compute configuration."""
        return self._config['compute']
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config['logging']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()