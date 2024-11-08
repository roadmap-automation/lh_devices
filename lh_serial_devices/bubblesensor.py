from uuid import uuid4

class BubbleSensorBase:
    """Base class for bubble sensor
    """

    def __init__(self, id: str | None = None, name: str = '') -> None:
        self.id = id if id is not None else str(uuid4())
        self.name = name

    async def initialize(self) -> None:
        """Initializes the bubble sensor (e.g. powers it on, connects to it)
        """

        raise NotImplementedError

    async def read(self) -> bool:
        """Reads the bubble sensor and returns its value

        Returns:
            bool: bubble sensor value
        """

        raise NotImplementedError
    
