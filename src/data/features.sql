-- BEHAVIORAL ASSUMPTIONS

-- table base with transactions' features.
CREATE TABLE transaction_features AS
SELECT *
FROM train_transaction t
LEFT JOIN train_identity i
USING ("TransactionID");

-- ==========================================
-- ---------- FINGERPRINTING USERS ----------
-- ----- 4 ways to recreate 'user_id' -------
-- ==========================================

-- uid1 = card1
ALTER TABLE transaction_features
ADD COLUMN uid1 text;

UPDATE transaction_features
SET uid1 = COALESCE(card1::text, 'missing');


-- uid2 = card1 + card2
ALTER TABLE transaction_features
ADD COLUMN uid2 text;

UPDATE transaction_features
SET uid2 =
    COALESCE(card1::text, 'missing') ||
    '_' ||
    COALESCE(card2::text, 'missing');


-- uid3 = card1 + card2 + addr1 + P_emaildomain
ALTER TABLE transaction_features
ADD COLUMN uid3 text;

UPDATE transaction_features
SET uid3 =
    COALESCE(card1::text, 'missing') ||
    '_' ||
    COALESCE(card2::text, 'missing') ||
    '_' ||
    COALESCE(addr1::text, 'missing') ||
    '_' ||
    COALESCE("P_emaildomain"::text, 'missing');


-- uid4 = card1 + card2 + addr1 + DeviceInfo
ALTER TABLE transaction_features
ADD COLUMN uid4 text;

UPDATE transaction_features
SET uid4 =
    COALESCE(card1::text, 'missing') ||
    '_' ||
    COALESCE(card2::text, 'missing') ||
    '_' ||
    COALESCE(addr1::text, 'missing') ||
    '_' ||
    COALESCE("DeviceInfo"::text, 'missing');

-- ==========================================
-- ------ Time features (window-based) ------
-- ==========================================

CREATE TABLE transaction_time_features AS
SELECT
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "DeviceInfo",
    "DeviceType",
    "uid1",
    "uid2",
    "uid3",
    "uid4",

    COUNT(*) OVER (
        PARTITION BY uid1 -- 1. GROUP BY uid1
        ORDER BY "TransactionDT" -- 2. sort by time
        RANGE BETWEEN 300 PRECEDING AND CURRENT ROW
    ) AS cnt_5m,

    COUNT(*) OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS cnt_1h,

    COUNT(*) OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS cnt_24h,

    COUNT(*) OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS cnt_7d,

    -- LAG: difference between two lines
    COALESCE("TransactionDT" - LAG("TransactionDT") OVER (PARTITION BY uid1 ORDER BY "TransactionDT"), 0
    ) AS time_since_last_tx,

    -- Amount sum last 1h
    SUM("TransactionAmt") OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS amt_1h,

    -- Per-user time-based-information (for preventing data leakage)
    COUNT(*)
        -- DATA LEAKAGE HANDLING
        OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    AS cnt_per_uid1,

    AVG("TransactionAmt")
        -- DATA LEAKAGE HANDLING
        OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    AS avg_amt_per_uid1, -- average transaction

    STDDEV("TransactionAmt")
        -- DATA LEAKAGE HANDLING
        OVER (
        PARTITION BY uid1
        ORDER BY "TransactionDT"
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    AS std_amt_per_uid1 -- std of transactions

FROM transaction_features;

-- Amount/Average Ratio
ALTER TABLE transaction_time_features
ADD COLUMN amt_vs_avg_ratio float8;

UPDATE transaction_time_features t
SET "amt_vs_avg_ratio" = ("TransactionAmt" / ("avg_amt_per_uid1" + 1));

-- =============================================
-- CTE time_since_last_geo_change (window-based)
-- =============================================

ALTER TABLE transaction_time_features
ADD COLUMN time_since_last_geo_change bigint;

-- MAX выдает 1 макс. значение из столбца, тогда как
-- MAX OVER запоминает макс. значение из предыдущих строчек:
-- 10   -> 10
-- NULL -> 10
-- NULL -> 10
-- 20   -> 20
-- NULL -> 20
-- MAX OVER PARTITION BY делает это для каждой подгруппы PARTITION.

-- slow algorithm:
-- UPDATE SET WITH step1, step2 - expensive operations are processed for each row.
-- new algorithm:
-- WITH step1, step 2 UPDATE SET FROM - step1, step 2 are executed only once.

-- 1. geo_change_dt = время смены адреса, либо null, если смены нет
-- 2. last_geo_change_dt - время последней смены адреса (считается через MAX OVER PARTITION)
-- 3. time_since_last_geo_change - разница между dt каждой транзакции и last_geo_change_dt

WITH step1 AS (
    SELECT
        "TransactionID",
        uid1,
        "TransactionDT",
        "addr1",
        CASE
            WHEN "addr1" != LAG("addr1") OVER (
                PARTITION BY uid1
                ORDER BY "TransactionDT"
            )
            THEN "TransactionDT"
            ELSE NULL
        END AS geo_change_dt -- (1)
    FROM transaction_features
),

step2 AS (
    SELECT
        *,
        MAX(geo_change_dt) OVER (
            PARTITION BY uid1
            ORDER BY "TransactionDT"
        ) AS last_geo_change_dt -- (2)
    FROM step1
)

UPDATE transaction_time_features
SET time_since_last_geo_change = step2."TransactionDT" - step2.last_geo_change_dt -- (3)
FROM step2
WHERE transaction_time_features."TransactionID" = step2."TransactionID";

-- =========================================
--  is_new_device
-- =========================================

UPDATE transaction_time_features
SET "DeviceInfo" = COALESCE("DeviceInfo", 'missing');

UPDATE transaction_time_features
SET "DeviceType" = COALESCE("DeviceType", 'missing');

ALTER TABLE transaction_time_features
ADD COLUMN is_new_device_uid1 bigint;

CREATE INDEX idx_device_uid1_lookup
ON transaction_time_features (
    "uid1",
    "DeviceInfo",
    "DeviceType",
    "TransactionDT"
);

UPDATE transaction_time_features t1
SET is_new_device_uid1 =
    CASE WHEN EXISTS(
        SELECT 1 FROM transaction_time_features t2
        WHERE t2."TransactionDT" < t1."TransactionDT"
        AND t2."DeviceInfo" = t1."DeviceInfo"
        AND t2."DeviceType" = t1."DeviceType"
        AND t2."uid1" = t1."uid1"
    )
    THEN 0 ELSE 1 END;

DROP INDEX idx_device_uid1_lookup;

-- =========================================
--  final_features
-- =========================================

-- SNIPPET: Get all columns names of the table as a list
SELECT string_agg(column_name, ', ' ORDER BY ordinal_position)
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'transaction_features';




CREATE TABLE final_features AS
SELECT t.*, f."isFraud", f."ProductCD", f."card1", f."card2", f."card3", f."card4", f."card5", f."card6", f."addr1", f."addr2", f."dist1", f."dist2", f."P_emaildomain", f."R_emaildomain", f."C1", f."C2", f."C3", f."C4", f."C5", f."C6", f."C7", f."C8", f."C9", f."C10", f."C11", f."C12", f."C13", f."C14", f."D1", f."D2", f."D3", f."D4", f."D5", f."D6", f."D7", f."D8", f."D9", f."D10", f."D11", f."D12", f."D13", f."D14", f."D15", f."M1", f."M2", f."M3", f."M4", f."M5", f."M6", f."M7", f."M8", f."M9", f."V1", f."V2", f."V3", f."V4", f."V5", f."V6", f."V7", f."V8", f."V9", f."V10", f."V11", f."V12", f."V13", f."V14", f."V15", f."V16", f."V17", f."V18", f."V19", f."V20", f."V21", f."V22", f."V23", f."V24", f."V25", f."V26", f."V27", f."V28", f."V29", f."V30", f."V31", f."V32", f."V33", f."V34", f."V35", f."V36", f."V37", f."V38", f."V39", f."V40", f."V41", f."V42", f."V43", f."V44", f."V45", f."V46", f."V47", f."V48", f."V49", f."V50", f."V51", f."V52", f."V53", f."V54", f."V55", f."V56", f."V57", f."V58", f."V59", f."V60", f."V61", f."V62", f."V63", f."V64", f."V65", f."V66", f."V67", f."V68", f."V69", f."V70", f."V71", f."V72", f."V73", f."V74", f."V75", f."V76", f."V77", f."V78", f."V79", f."V80", f."V81", f."V82", f."V83", f."V84", f."V85", f."V86", f."V87", f."V88", f."V89", f."V90", f."V91", f."V92", f."V93", f."V94", f."V95", f."V96", f."V97", f."V98", f."V99", f."V100", f."V101", f."V102", f."V103", f."V104", f."V105", f."V106", f."V107", f."V108", f."V109", f."V110", f."V111", f."V112", f."V113", f."V114", f."V115", f."V116", f."V117", f."V118", f."V119", f."V120", f."V121", f."V122", f."V123", f."V124", f."V125", f."V126", f."V127", f."V128", f."V129", f."V130", f."V131", f."V132", f."V133", f."V134", f."V135", f."V136", f."V137", f."V138", f."V139", f."V140", f."V141", f."V142", f."V143", f."V144", f."V145", f."V146", f."V147", f."V148", f."V149", f."V150", f."V151", f."V152", f."V153", f."V154", f."V155", f."V156", f."V157", f."V158", f."V159", f."V160", f."V161", f."V162", f."V163", f."V164", f."V165", f."V166", f."V167", f."V168", f."V169", f."V170", f."V171", f."V172", f."V173", f."V174", f."V175", f."V176", f."V177", f."V178", f."V179", f."V180", f."V181", f."V182", f."V183", f."V184", f."V185", f."V186", f."V187", f."V188", f."V189", f."V190", f."V191", f."V192", f."V193", f."V194", f."V195", f."V196", f."V197", f."V198", f."V199", f."V200", f."V201", f."V202", f."V203", f."V204", f."V205", f."V206", f."V207", f."V208", f."V209", f."V210", f."V211", f."V212", f."V213", f."V214", f."V215", f."V216", f."V217", f."V218", f."V219", f."V220", f."V221", f."V222", f."V223", f."V224", f."V225", f."V226", f."V227", f."V228", f."V229", f."V230", f."V231", f."V232", f."V233", f."V234", f."V235", f."V236", f."V237", f."V238", f."V239", f."V240", f."V241", f."V242", f."V243", f."V244", f."V245", f."V246", f."V247", f."V248", f."V249", f."V250", f."V251", f."V252", f."V253", f."V254", f."V255", f."V256", f."V257", f."V258", f."V259", f."V260", f."V261", f."V262", f."V263", f."V264", f."V265", f."V266", f."V267", f."V268", f."V269", f."V270", f."V271", f."V272", f."V273", f."V274", f."V275", f."V276", f."V277", f."V278", f."V279", f."V280", f."V281", f."V282", f."V283", f."V284", f."V285", f."V286", f."V287", f."V288", f."V289", f."V290", f."V291", f."V292", f."V293", f."V294", f."V295", f."V296", f."V297", f."V298", f."V299", f."V300", f."V301", f."V302", f."V303", f."V304", f."V305", f."V306", f."V307", f."V308", f."V309", f."V310", f."V311", f."V312", f."V313", f."V314", f."V315", f."V316", f."V317", f."V318", f."V319", f."V320", f."V321", f."V322", f."V323", f."V324", f."V325", f."V326", f."V327", f."V328", f."V329", f."V330", f."V331", f."V332", f."V333", f."V334", f."V335", f."V336", f."V337", f."V338", f."V339", f."id_01", f."id_02", f."id_03", f."id_04", f."id_05", f."id_06", f."id_07", f."id_08", f."id_09", f."id_10", f."id_11", f."id_12", f."id_13", f."id_14", f."id_15", f."id_16", f."id_17", f."id_18", f."id_19", f."id_20", f."id_21", f."id_22", f."id_23", f."id_24", f."id_25", f."id_26", f."id_27", f."id_28", f."id_29", f."id_30", f."id_31", f."id_32", f."id_33", f."id_34", f."id_35", f."id_36", f."id_37", f."id_38"
FROM transaction_time_features t
LEFT JOIN transaction_features AS f ON t."TransactionID" = f."TransactionID"
