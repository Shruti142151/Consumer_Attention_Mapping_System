class ProductScoringEngine:

    def __init__(self):
        self.product_scores = {}


    def calculate_score(
        self,
        product_id,
        attention_duration,
        interaction_frequency,
        pickup_rate,
        conversion_rate,
        repeat_engagement
    ):

        score = (
            attention_duration * 0.35
            +
            interaction_frequency * 0.25
            +
            pickup_rate * 0.20
            +
            conversion_rate * 0.15
            +
            repeat_engagement * 0.05
        )


        self.product_scores[product_id] = {
            "attention_score": attention_duration,
            "interaction_score": interaction_frequency,
            "pickup_score": pickup_rate,
            "conversion_score": conversion_rate,
            "repeat_score": repeat_engagement,
            "attractiveness_score": score
        }


        return score


    def get_product_score(self, product_id):

        return self.product_scores.get(
            product_id,
            None
        )