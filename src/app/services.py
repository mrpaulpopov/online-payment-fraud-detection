import json
import time
from fastapi import APIRouter, Depends, HTTPException, Response, status, Request

import yaml
from fastapi import HTTPException
import logging
from src.paths import INFERENCE_PATH
from src.pipelines.inference_pipeline import inference_pipeline


def apply_business_rules(transaction: dict) -> tuple[bool, str]:
    if transaction['TransactionAmt'] > 500000 and transaction['is_new_device_uid1'] == 1:
        return True, "Blocked by Business Rule: Huge amount from new device"

    return False, ""


def graceful_degradation(transaction: dict) -> bool:
    if transaction['TransactionAmt'] > 10_000_000:
        return True

    return False


def process_payment(transaction: dict, inference_meta, model_lgbm) -> tuple[str, bool, float, str]:
    fraud_probability = None
    is_fraud = None
    business_decision, rule_reason = apply_business_rules(transaction)

    if business_decision is True:
        reason = f"Fraud (Blocked by Business Rules: {rule_reason})"
        return str(transaction["TransactionID"]), True, fraud_probability, reason

    try:
        fraud_probability, is_fraud = inference_pipeline(transaction, inference_meta, model_lgbm)
        if is_fraud:
            reason = f"Fraud (Blocked by ML, probability: {(fraud_probability[0])*100:.1f}%)"
        else:
            reason = "Legit (Passed ML)"
    except Exception as e:
        logging.error(f"ML Pipeline failed: {str(e)}. Falling back to Graceful Degradation.")
        is_fraud = graceful_degradation(transaction)
        if is_fraud is True:
            reason = "Fraud (Blocked by Fallback rules)"
        else:
            reason = "Legit (Passed Fallback rules)"

    if fraud_probability is not None:
        fraud_probability = round(fraud_probability[0], 4)
    return str(transaction["TransactionID"]), is_fraud, fraud_probability, reason
