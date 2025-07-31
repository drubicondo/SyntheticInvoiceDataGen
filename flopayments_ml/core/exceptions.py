class EnvironmentConfigError(Exception):
    """Raised when environment variables cannot be loaded or are missing"""
    pass

class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

class GenerationError(Exception):
    """Raised when data generation fails"""
    pass


class ContentFilterError(Exception):
    """Raised when a response is blocked by the OpenAI content filter"""
    pass
