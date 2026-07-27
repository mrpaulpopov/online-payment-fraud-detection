import json
import time
from fastapi import APIRouter, Depends, HTTPException, Response, status, Request

import yaml
from fastapi import HTTPException
import logging
from src.paths import INFERENCE_PATH
from src.pipelines.inference_pipeline import inference_pipeline


def apply_business_rules(transaction: dict):
    if transaction['TransactionAmt'] > 500000 and transaction['is_new_device_uid1'] == 1:
        return True, "Blocked by Business Rule: Huge amount from new device"

    return False, ""


def graceful_degradation(transaction: dict):
    if transaction['TransactionAmt'] > 10_000_000:
        return True

    return False


def process_payment(input_transaction: dict, inference_meta, model_lgbm):
    fraud_probability = None
    is_fraud = None
    business_decision, rule_reason = apply_business_rules(input_transaction)

    if business_decision is True:
        reason = f"Fraud (Blocked by Business Rules: {rule_reason})"
        # return transaction_id, fraud_probability, is_fraud, reason # TODO
        return True, fraud_probability, reason

    try:
        # features = extract_features(transaction) # TODO
        fraud_probability, is_fraud = inference_pipeline(input_transaction, inference_meta, model_lgbm)
        if is_fraud:
            reason = f"Fraud (Blocked by ML, probability: {(fraud_probability[0])*100:.1f}%)"
        else:
            reason = "Legit (Passed ML)"
    except Exception as e:
        logging.error(f"ML Pipeline failed: {str(e)}. Falling back to Graceful Degradation.")
        is_fraud = graceful_degradation(input_transaction)
        if is_fraud is True:
            reason = "Fraud (Blocked by Fallback rules)"
        else:
            reason = "Legit (Passed Fallback rules)"

    # return transaction_id, fraud_probability, is_fraud, reason # TODO
    if fraud_probability is not None:
        fraud_probability = round(fraud_probability[0], 4)
    return is_fraud, fraud_probability, reason
