"""
Simple Logger Stub
=================
Basic logger implementation if needed.
"""

import logging

class Logger:
    """Simple logger wrapper"""
    
    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
