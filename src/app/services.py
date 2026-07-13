import json
import time
from fastapi import APIRouter, Depends, HTTPException, Response, status, Request

import yaml
from fastapi import HTTPException
import logging
from src.paths import INFERENCE_PATH
from src.pipelines.inference_pipeline import inference_pipeline


def apply_business_rules(transaction):
    if transaction['amount'] > 500000 and transaction['is_new_device']:
        return True, "Blocked by Rule: Huge amount from new device"

    return None, "Pass to ML"


def process_payment(input_transaction, inference_meta, num_imputer, scaler, model_lgbm, model_pytorch):

    fraud_probability = None
    is_fraud = None
    rule_decision, rule_reason = apply_business_rules(input_transaction)

    if rule_decision is True:
        action = f"Fraud (Blocked by Rules: {rule_reason})"
        # return transaction_id, fraud_probability, is_fraud, action # TODO

    # features = extract_features(transaction) # TODO

    fraud_probability, is_fraud = inference_pipeline(input_transaction, inference_meta,
                                                     num_imputer, scaler,
                                                     model_lgbm, model_pytorch)

    if is_fraud:
        action = f"Fraud (Blocked by ML, confidence: {fraud_probability})"
    else:
        action = "Legit (Passed ML)"

    # return transaction_id, fraud_probability, is_fraud, action # TODO
    return fraud_probability, is_fraud, action
