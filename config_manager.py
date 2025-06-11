import json
import os
from typing import Any, Dict, Optional

class ConfigManager:
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default if not exists"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {str(e)}")
                return self._create_default_config()
        else:
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            'source_directory': '',
            'target_directory': '',
            'container_directory': '',
            'enable_ai': True,
            'zero_change_mode': False,
            'similarity_threshold': 0.8,
            'document_types': {
                'invoice': ['invoice', 'inv', 'bill'],
                'packing_list': ['packing', 'packing list', 'pl'],
                'bill_of_lading': ['bl', 'bill of lading', 'b/l'],
                'certificate': ['certificate', 'cert', 'ce'],
                'manual': ['manual', 'guide', 'instructions'],
                'price_list': ['price', 'pricelist', 'pricing']
            }
        }

        # Save default config
        self.save_config(default_config)

        return default_config

    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file"""
        if config is not None:
            self.config = config

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
        self.save_config()

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config.update(config_dict)
        self.save_config()

    def reset(self) -> None:
        """Reset configuration to default values"""
        self.config = self._create_default_config()
        self.save_config()