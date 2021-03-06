from rasa.nlu.components import Component
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer(Component):
    """A pre-trained sentiment component"""

    name = "sentiment"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["it"]

    def __init__(self, component_config=None):
        super(SentimentAnalyzer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

        return entity

    def process(self, message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""

        sid = SentimentIntensityAnalyzer()
        try:
            res = sid.polarity_scores(message.data['text'])
        except KeyError:
            return[]

        compound = res['compound']
        del res['compound']
        key, value = max(res.items(), key=lambda x: x[1])
        if compound >= 0.05:
            key = "Positive"
        elif compound <= - 0.05:
            key = "Negative"
        else:
            key = "Neutral"

        entity = self.convert_to_rasa(key, value)

        message.set("entities", [entity], add_to_output=True)

    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""

        pass