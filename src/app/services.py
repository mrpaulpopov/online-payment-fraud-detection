import logging

import lightgbm as lgb

from src.pipelines.inference_pipeline import inference_pipeline

logger = logging.getLogger(__name__)

def apply_business_rules(transaction: dict) -> tuple[bool, str]:
    if transaction['TransactionAmt'] > 500000 and transaction['is_new_device_uid1'] == 1:
        return True, "Blocked by Business Rule: Huge amount from new device"

    return False, ""


def graceful_degradation(transaction: dict) -> bool:
    return transaction['TransactionAmt'] > 10_000_000


def process_payment(transaction: dict, inference_meta: dict, model_lgbm: lgb.Booster) -> tuple[str, bool, float | None, str]:
    fraud_probability: float | None = None
    is_fraud: bool = False
    business_decision, rule_reason = apply_business_rules(transaction)

    if business_decision is True:
        reason = f"Fraud (Blocked by Business Rules: {rule_reason})"
        return str(transaction["TransactionID"]), True, fraud_probability, reason

    try:
        raw_fraud_probability, raw_is_fraud = inference_pipeline(transaction, inference_meta, model_lgbm) # np.array
        fraud_probability = float(raw_fraud_probability[0]) # np.array to float
        is_fraud = bool(raw_is_fraud[0])

        if is_fraud:
            reason = f"Fraud (Blocked by ML, probability: {fraud_probability*100:.1f}%)"
        else:
            reason = "Legit (Passed ML)"
    except Exception as e: # noqa: BLE001
        logger.error(f"ML Pipeline failed: {e}. Falling back to Graceful Degradation.")
        is_fraud = bool(graceful_degradation(transaction))
        if is_fraud is True:
            reason = "Fraud (Blocked by Fallback rules)"
        else:
            reason = "Legit (Passed Fallback rules)"

    if fraud_probability is not None:
        fraud_probability = round(fraud_probability, 4)
    return str(transaction["TransactionID"]), is_fraud, fraud_probability, reason
