import json
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from api_manager import api_manager

@dataclass
class AutoSelectionConfig:
    """Auto-selection configuration"""
    enabled: bool = True
    crypto_count: int = 20
    stock_count: int = 20
    refresh_hours: int = 24
    selection_strategy: str = "mixed_portfolio"  # mixed_portfolio, market_cap, trending
    exclude_cryptos: List[str] = field(default_factory=list)
    exclude_stocks: List[str] = field(default_factory=list)

@dataclass
class TradingConfig:
    """Trading configuration"""
    risk_per_trade: float = 0.01
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    risk_free_rate: float = 0.02

@dataclass
class AppConfig:
    """Application configuration"""
    crypto_list: List[str] = field(default_factory=lambda: ["bitcoin", "ethereum"])
    stock_list: List[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    update_interval: int = 300
    trading: TradingConfig = field(default_factory=TradingConfig)
    auto_selection: AutoSelectionConfig = field(default_factory=AutoSelectionConfig)
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    last_asset_update: str = ""

class ConfigManager:
    """Enhanced configuration management with auto-selection"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> AppConfig:
        """Load configuration with auto-selection support"""
        if not self.config_file.exists():
            self.logger.warning(f"Config file {self.config_file} not found, using defaults")
            return AppConfig()
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            # Parse configurations
            trading_data = data.get('trading', {})
            trading_config = TradingConfig(**trading_data)
            
            auto_selection_data = data.get('auto_selection', {})
            auto_selection_config = AutoSelectionConfig(**auto_selection_data)
            
            return AppConfig(
                crypto_list=data.get('crypto_list', AppConfig().crypto_list),
                stock_list=data.get('stock_list', AppConfig().stock_list),
                update_interval=data.get('update_interval', 300),
                trading=trading_config,
                auto_selection=auto_selection_config,
                strategy_params=data.get('strategy_params', {}),
                last_asset_update=data.get('last_asset_update', "")
            )
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return AppConfig()
    
    def save_config(self):
        """Save configuration with auto-selection data"""
        try:
            config_dict = {
                'crypto_list': self.config.crypto_list,
                'stock_list': self.config.stock_list,
                'update_interval': self.config.update_interval,
                'trading': {
                    'risk_per_trade': self.config.trading.risk_per_trade,
                    'stop_loss_pct': self.config.trading.stop_loss_pct,
                    'take_profit_pct': self.config.trading.take_profit_pct,
                    'risk_free_rate': self.config.trading.risk_free_rate
                },
                'auto_selection': {
                    'enabled': self.config.auto_selection.enabled,
                    'crypto_count': self.config.auto_selection.crypto_count,
                    'stock_count': self.config.auto_selection.stock_count,
                    'refresh_hours': self.config.auto_selection.refresh_hours,
                    'selection_strategy': self.config.auto_selection.selection_strategy,
                    'exclude_cryptos': self.config.auto_selection.exclude_cryptos,
                    'exclude_stocks': self.config.auto_selection.exclude_stocks
                },
                'strategy_params': self.config.strategy_params,
                'last_asset_update': self.config.last_asset_update
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def update_assets_from_auto_selection(self, selected_assets: Dict[str, Any]):
        """Update configuration with auto-selected assets"""
        try:
            self.config.crypto_list = selected_assets['crypto_list']
            self.config.stock_list = selected_assets['stock_list']
            self.config.last_asset_update = selected_assets['last_updated']
            self.save_config()
            
            self.logger.info(f"Updated assets: {len(self.config.crypto_list)} cryptos, {len(self.config.stock_list)} stocks")
        except Exception as e:
            self.logger.error(f"Error updating assets from auto-selection: {e}")
    
    def should_refresh_assets(self) -> bool:
        """Check if assets should be auto-refreshed"""
        if not self.config.auto_selection.enabled:
            return False
        
        if not self.config.last_asset_update:
            return True
        
        try:
            last_update = datetime.fromisoformat(self.config.last_asset_update)
            time_diff = datetime.now() - last_update
            return time_diff.total_seconds() > (self.config.auto_selection.refresh_hours * 3600)
        except:
            return True
    
    def get_api_key(self, service: str) -> str:
        """Get API key for service"""
        return api_manager.get_key(service)
