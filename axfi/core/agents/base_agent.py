"""
Base Agent Class for AXFI
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Base class for all AXFI agents.
    Provides common initialization and interface.
    """
    
    def __init__(self, config: dict, **kwargs):
        """
        Initialize agent with configuration.
        
        Args:
            config: System configuration dictionary
            **kwargs: Additional arguments for specific agents
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the agent's main task.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            Dictionary with results
        """
        raise NotImplementedError("Subclass must implement run()")
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if valid
        """
        return True
    
    def log_result(self, result: Dict[str, Any]):
        """
        Log agent result.
        
        Args:
            result: Result dictionary
        """
        self.logger.info(f"{self.__class__.__name__} completed: {result.get('status', 'unknown')}")


if __name__ == "__main__":
    # Test base agent
    class TestAgent(BaseAgent):
        def run(self, **kwargs):
            return {"status": "success", "message": "Test agent executed"}
    
    config = {"test": True}
    agent = TestAgent(config)
    result = agent.run()
    print(f"Test result: {result}")

