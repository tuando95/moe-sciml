import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class Config:
    """Configuration management for AME-ODE."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yml"
        
        self._config = self._load_config_with_includes(config_path)
    
    def _load_config_with_includes(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML config file, handling !include directives."""
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Process includes
        base_dir = config_path.parent
        lines = content.split('\n')
        
        # Find and process !include directives
        include_pattern = re.compile(r'^\s*!include\s+(.+)$')
        processed_content = []
        
        for line in lines:
            match = include_pattern.match(line)
            if match:
                include_file = match.group(1).strip()
                include_path = base_dir / include_file
                if include_path.exists():
                    # Load the included file
                    included_config = self._load_config_with_includes(include_path)
                    # We'll merge this after loading the current file
                    # For now, just skip the include line
                    continue
            processed_content.append(line)
        
        # Load the current file without includes
        current_config = yaml.safe_load('\n'.join(processed_content))
        if current_config is None:
            current_config = {}
        
        # Now process includes and merge
        for line in lines:
            match = include_pattern.match(line)
            if match:
                include_file = match.group(1).strip()
                include_path = base_dir / include_file
                if include_path.exists():
                    included_config = self._load_config_with_includes(include_path)
                    # Deep merge: included config is the base, current config overrides
                    current_config = self._deep_merge(included_config, current_config)
        
        return current_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
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