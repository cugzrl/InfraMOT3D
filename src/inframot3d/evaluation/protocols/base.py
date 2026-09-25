class BaseProtocol:
    name = "base"

    def normalize_class(self, class_name):
        raise NotImplementedError

    def filter_gt(self, objects):
        raise NotImplementedError

    def filter_prediction(self, objects):
        raise NotImplementedError

    def iou_threshold(self):
        raise NotImplementedError

    def get_eval_classes(self):
        raise NotImplementedError

    def score_sampling(self):
        raise NotImplementedError

    def aggregate_metrics(self, raw):
        raise NotImplementedError
