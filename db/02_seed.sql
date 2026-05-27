COPY test_identity FROM '/data/test_identity.csv' WITH (FORMAT csv, HEADER true);
COPY test_transaction FROM '/data/test_transaction.csv' WITH (FORMAT csv, HEADER true);
COPY train_identity FROM '/data/train_identity.csv' WITH (FORMAT csv, HEADER true);
COPY train_transaction FROM '/data/train_transaction.csv' WITH (FORMAT csv, HEADER true);