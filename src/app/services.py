from src.ml.predict import predict_user_items
from fastapi import HTTPException
import logging



def apply_business_rules(transaction):
    if transaction['amount'] > 500000 and transaction['is_new_device']:
        return True, "Blocked by Rule: Huge amount from new device"

    return None, "Pass to ML"


def process_payment(transaction):
    rule_decision, rule_reason = apply_business_rules(transaction)

    if rule_decision is True:
        return "Fraud (Blocked by Rules)"


    features = extract_features(transaction)
    prob = fraud_model.predict(features)[0]

    if prob > best_threshold:
        return f"Fraud (Blocked by ML, confidence: {prob})"
    else:
        return "Legit (Passed ML)"
