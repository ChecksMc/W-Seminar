from abc import ABC, abstractmethod
import numpy as np



    
class Image(ABC):
    """ 
    Represents an image with associated emotion data. 
    - location is a string representation of the image's file path.
    - emotion_data is a numpy array representing the emotion associated with the image using a 3D vector.
    - data_type is a string indicating the type. Supported Types: ("RGB", "HSV", "LAB")
    """
    location : str
    emotion_data : np.ndarray
    data_type : str
    @abstractmethod
    def get_image(self):
        raise NotImplementedError()

    @abstractmethod
    def get_emotion_data(self):
        raise NotImplementedError()


class Analyser(ABC):
    data : list[Image]
    @abstractmethod
    def analyse(self):
        raise NotImplementedError()

