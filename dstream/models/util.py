from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = 'Classification'
    REGRESSION = 'Regresssion'
    CLUSTERING = 'Clustering'