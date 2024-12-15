from river.metrics import Accuracy
from AutoML_pipeline.Change_pipeline import change_pipeline
from AutoML_pipeline.Find_best_pipeline_ProcessPoolExecutor import find_best_pipeline
from Functions.Accuracy_check import accuracy_check
from Functions.Split_data import split_data
from river import base, drift
from Functions.Sliding_Window_Class import SlidingWindow
import time


class AutomlPipeline(base.Classifier):

    def __init__(self, target, data_drift_detector, concept_drift_detector, seed: int | None = None):
        self.target = target
        self.seed = seed
        if data_drift_detector:
            self.data_drift_detector_method = drift.ADWIN()

        else:
            self.data_drift_detector_method = None

        if concept_drift_detector:
            self.concept_drift_detector_method = drift.ADWIN()

        else:
            self.concept_drift_detector_method = None

        # set a buffer size
        self.buffer_size = 450

        self.k = None  # index for the buffers

        # create a buffer to save the latest feature data
        self.x_buffer = SlidingWindow(self.buffer_size)

        # create a buffer to save the latest target data
        self.y_buffer = SlidingWindow(self.buffer_size)

        # create the instance of metric
        self.accuracy = Accuracy()

        # create a list to save the concept drifts
        self.concept_drifts = []

        # create a list to save the data drifts
        self.data_drifts = []

        # create a buffer to save the predicted values until the real value comes
        self.y_predicted_buffer_temp = []

        # create a buffer to save the predicted
        self.y_predicted_buffer = SlidingWindow(self.buffer_size)

        # initialize some useful variables for drift detection
        self.distance_from_last_data_drift_detected = None
        self.latest_data_drift_index = None
        self.latest_data_drift_feature = None
        self.distance_from_last_consept_drift_detected = None

        # the last point where the model changed
        self.model_change_point = self.buffer_size

        # flag that helps us to know if a model need retrain or not - False = Not need retrain, True = Need retrain
        self.need_retrain = False

        self.pipeline = None

        self.data_drift_detector = None

        self.concept_drift_detector = None

        self.index = 0

    def init_train(self, init_train_data):
        start_time = time.time()
        # crate list to save the dataset splitted to features and target, for the initial train
        x_train = []
        y_train = []
        self.index = len(init_train_data)

        for instance in init_train_data:
            x, y = split_data(instance, self.target)
            x_train.append(x)
            y_train.append(y)

        # find the best pipeline

        self.pipeline, _, self.data_drift_detector, self.concept_drift_detector = (
            find_best_pipeline(x_train, y_train, self.data_drift_detector_method, self.concept_drift_detector_method, self.seed))

        end_time = time.time()
        total_training_time = end_time - start_time
        print("Pipeline selected: \n", self.pipeline)
        print(f"Total initial training time: {total_training_time:.2f} seconds")
        print("-" * 10)

    def predict_one(self, x: dict, **kwargs):

        y = self.pipeline.predict_one(x, **kwargs)  # make a prediction with the pipeline
        self.y_predicted_buffer_temp.append(y)

        return y  # return the prediction

    def learn_one(self, x: dict, y):

        self.index += 1
        self.x_buffer.add(x)
        self.y_predicted_buffer.add(self.y_predicted_buffer_temp.pop(0))

        if self.data_drift_detector_method is not None:

            for feature, value in x.items():
                self.data_drift_detector[feature].update(value)
                if self.data_drift_detector[feature].drift_detected:
                    # I don't start to look for a data drift if there is already detected a concept drift
                    if self.distance_from_last_consept_drift_detected is None:
                        # find the real point of data drift
                        temp_distance = self.data_drift_detector[feature].width
                        if self.distance_from_last_data_drift_detected is None or temp_distance < self.distance_from_last_data_drift_detected:
                            self.distance_from_last_data_drift_detected = temp_distance
                        if self.distance_from_last_data_drift_detected > self.buffer_size:
                            self.distance_from_last_data_drift_detected = self.buffer_size
                        self.latest_data_drift_index = self.index
                        self.latest_data_drift_feature = feature
                    break
        if (self.distance_from_last_data_drift_detected is not None
                and self.distance_from_last_data_drift_detected <= self.buffer_size):
            if self.distance_from_last_data_drift_detected == self.buffer_size:
                # check if we have accuracy drop more than 5%
                self.need_retrain = accuracy_check(self.accuracy, self.y_buffer.get(),
                                                   self.y_predicted_buffer.get(), 0.07)

                if self.need_retrain:
                    print(f"Data drift detected at data point {self.latest_data_drift_index} "
                          f"in feature {self.latest_data_drift_feature}")
                    self.data_drifts.append(self.latest_data_drift_index)

            self.distance_from_last_data_drift_detected += 1

        self.y_buffer.add(y)
        # update the accuracy
        self.accuracy.update(y, self.y_predicted_buffer.get_specific(-1))
        self.pipeline.learn_one(x, y)

        if self.concept_drift_detector_method is not None:
            self.concept_drift_detector.update(self.y_predicted_buffer.get_specific(-1) != y)

            # Check if change was detected
            if self.concept_drift_detector.drift_detected:
                print(f"Concept drift detected at data point {self.index}")

                self.concept_drifts.append(self.index)

                # find the real point of concept drift
                self.distance_from_last_consept_drift_detected = self.concept_drift_detector.width
                if (self.distance_from_last_data_drift_detected is not None and
                        self.distance_from_last_consept_drift_detected > self.distance_from_last_data_drift_detected):
                    self.distance_from_last_consept_drift_detected = self.distance_from_last_data_drift_detected
                if self.distance_from_last_consept_drift_detected > self.buffer_size:
                    self.distance_from_last_consept_drift_detected = self.buffer_size

            if (self.distance_from_last_consept_drift_detected is not None
                    and self.distance_from_last_consept_drift_detected <= self.buffer_size):
                if self.distance_from_last_consept_drift_detected == self.buffer_size:
                    self.need_retrain = True

                self.distance_from_last_consept_drift_detected += 1

        # retrain
        if self.need_retrain:
            start_time = time.time()

            buffer_accuracy = Accuracy()

            for j in range(self.buffer_size):
                buffer_accuracy.update(self.y_buffer.get_specific(j), self.y_predicted_buffer.get_specific(j))

            self.pipeline, self.accuracy, self.data_drift_detector, self.concept_drift_detector \
                = change_pipeline(self.pipeline, self.x_buffer.get(), self.y_buffer.get(), self.data_drift_detector_method,
                                  self.concept_drift_detector_method, buffer_accuracy, self.seed)

            self.distance_from_last_data_drift_detected = None
            self.distance_from_last_consept_drift_detected = None
            self.model_change_point = self.index
            self.need_retrain = False

            end_time = time.time()
            total_retraining_time = end_time - start_time
            print(f"Total retraining time: {total_retraining_time:.2f} seconds")
            print("-" * 10)
