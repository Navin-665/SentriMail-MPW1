"""
All hyperparameters and paths for training the SentriMail multi-task
model, kept in one place so nothing needs hunting through dataset.py,
model.py, or train.py to find a number that needs changing.
"""

import os

# ---- paths ----
PROCESSED_DATA_DIR = "data/processed"
TRAIN_PARQUET_PATH = os.path.join(PROCESSED_DATA_DIR, "train.parquet")
VAL_PARQUET_PATH = os.path.join(PROCESSED_DATA_DIR, "val.parquet")
TEST_PARQUET_PATH = os.path.join(PROCESSED_DATA_DIR, "test.parquet")
ECOMMERCE_VALIDATION_PATH = os.path.join(PROCESSED_DATA_DIR, "validation_ecommerce.parquet")

CHECKPOINT_DIR = "checkpoints"

# ---- backbone ----
BACKBONE_MODEL_NAME = "t5-small"

# ---- label vocabularies ----
# fixed order matters - index 0 always means the same label everywhere,
# from dataset.py through model.py through evaluate.py
SENTIMENT_LABELS = ["negative", "positive"]
EMOTION_LABELS = ["anger", "fear", "sadness", "disgust", "surprise", "joy", "neutral"]
CATEGORY_LABELS = ["billing", "technical", "delivery", "customer_service", "product", "refund", "other"]

# ---- tokenization ----
MAX_INPUT_TOKENS = 256      # complaint text - long CFPB narratives get truncated past this
MAX_RESPONSE_TOKENS = 128   # generated responses are meant to be concise, not essays

# ---- training ----
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

# ---- class weighting ----
# "class_balanced" fixes a real, confirmed problem: plain inverse-frequency
# gave billing (huge, common) and other (medium, broad) a ~14x weight gap,
# pushing the model to sacrifice billing accuracy to chase rare-looking
# classes - billing recall was 0.70 vs 0.97 precision, the signature of
# systematic under-prediction. class_balanced brings that gap to ~1:1.
CLASS_WEIGHT_METHOD = "class_balanced"
CLASS_WEIGHT_BETA = 0.999

# ---- confidence threshold ----
CONFIDENCE_THRESHOLD = 0.65  # low-confidence category routing flagged for review

# ---- multi-task loss weighting ----
# generation gets the heaviest weight deliberately - it's the hardest task
# and has the least real training signal (only ~132k of 338k rows have an
# actual target response), so left unweighted it would get drowned out by
# the three classification losses, which have signal on every row
LOSS_WEIGHT_SENTIMENT = 1.0
LOSS_WEIGHT_EMOTION = 1.0
LOSS_WEIGHT_CATEGORY = 1.0
LOSS_WEIGHT_GENERATION = 2.0

# ---- misc ----
CHECKPOINT_EVERY_STEPS = 500   # save a resumable checkpoint every N training steps
RANDOM_SEED = 42
