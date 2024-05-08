# Databricks notebook source
import os
import io
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# COMMAND ----------

# AWS credentials
access_key = 'NEEDS TO BE INPUT'
secret_key = 'NEEDS TO BE INPUT'
aws_region = "us-east-1"
bucket_name = 'ham10000skincancer'

# Initialize the S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=aws_region
)

# Function to read CSV file from S3
def read_csv_file(key):
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(obj['Body'])

# Function to read an image from S3
def read_image(key):
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    img = Image.open(io.BytesIO(obj['Body'].read()))
    return img

# Function to find the image path on S3
def find_image_path(image_id):
    possible_paths = [
        f"skin-cancer-mnist-ham10000/HAM10000_images_part_1/{image_id}.jpg",
        f"skin-cancer-mnist-ham10000/HAM10000_images_part_2/{image_id}.jpg"
    ]
    for path in possible_paths:
        try:
            s3_client.head_object(Bucket=bucket_name, Key=path)
            return path  # Path exists
        except s3_client.exceptions.ClientError:
            continue
    return None  # Path not found

# Setup metadata for the dataset 3000MAX
def setup_metadata(limit=3000):
    key = 'skin-cancer-mnist-ham10000/HAM10000_metadata.csv'
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        df_metadata = pd.read_csv(io.BytesIO(obj['Body'].read()))

        if limit and len(df_metadata) > limit:
            df_metadata = df_metadata.sample(n=limit, random_state=42).reset_index(drop=True)

        df_metadata['s3_image_path'] = df_metadata['image_id'].apply(find_image_path)

        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        df_metadata['cell_type'] = df_metadata['dx'].map(lesion_type_dict.get)
        df_metadata['cell_type_idx'] = pd.Categorical(df_metadata['cell_type']).codes
        
        print("Metadata setup completed. Processed {} entries.".format(len(df_metadata)))
        return df_metadata
    except Exception as e:
        print(f"Error setting up metadata: {str(e)}")
        return None

df_metadata = setup_metadata()

# COMMAND ----------

# Cache loaded images
loaded_images_cache = {}

def load_image_from_s3_and_cache(bucket, key):
    if key in loaded_images_cache:
        return loaded_images_cache[key]
    else:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()
        image = Image.open(io.BytesIO(image_content)).convert('RGB')
        image = image.resize((28, 28))
        image_np = np.array(image) / 255.0
        loaded_images_cache[key] = image_np
        return image_np

# Data pipeline architecture to stream images
def image_stream_from_s3():
    df_metadata = setup_metadata()  # Load metadata
    for _, row in df_metadata.iterrows():
        if row['s3_image_path']:
            image_np = load_image_from_s3_and_cache(bucket_name, row['s3_image_path'])
            label = row['cell_type_idx']  # Calculate label from metadata
            yield (image_np, label)

# Create TensorFlow datasets from image stream
def create_tensorflow_datasets_from_stream():
    image_stream = image_stream_from_s3()
    dataset = tf.data.Dataset.from_generator(lambda: image_stream, output_types=(tf.float32, tf.int32))
    # Split the dataset into train and validation sets
    train_ds, val_ds = dataset.take(int(0.8 * len(df_metadata))), dataset.skip(int(0.8 * len(df_metadata)))
    # Shuffle and batch the datasets
    train_ds = train_ds.shuffle(1000).batch(32)
    val_ds = val_ds.batch(32)
    return train_ds, val_ds

df_metadata = setup_metadata()  # Load metadata
train_ds, val_ds = create_tensorflow_datasets_from_stream()
print("Datasets created successfully, ready to train the model.")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count

# Setup metadata and prepare datasets
df_metadata = setup_metadata()
if df_metadata is not None:
    # Convert df_metadata to a Spark DataFrame
    df_metadata_spark = spark.createDataFrame(df_metadata)
    # Register the DataFrame as a SQL temporary view
    df_metadata_spark.createOrReplaceTempView("metadata")
    print("DataFrame registered as a SQL temporary view.")


# Metadata and Data Integrity Check
null_checks = df_metadata_spark.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_metadata_spark.columns])
null_checks.show()

# Check columns for nulls and raise errors if necessary
cell_type_idx_nulls = spark.sql("SELECT count(*) FROM metadata WHERE cell_type_idx IS NULL")
s3_image_path_nulls = spark.sql("SELECT count(*) FROM metadata WHERE s3_image_path IS NULL")

cell_type_idx_null_count = cell_type_idx_nulls.collect()[0][0]
s3_image_path_null_count = s3_image_path_nulls.collect()[0][0]

if cell_type_idx_null_count > 0:
    raise ValueError("Missing values found in 'cell_type_idx'.")
if s3_image_path_null_count > 0:
    raise ValueError("Missing image paths.")

print(f"Number of missing image paths: {s3_image_path_null_count}")
print(f"Total number of images in metadata: {df_metadata_spark.count()}")
print(f"Percentage of missing paths: {s3_image_path_null_count / df_metadata_spark.count() * 100:.2f}%")
print("Metadata checks passed successfully.")

# Data Exploration / Visual check of distribution of classes
spark.sql("""
SELECT cell_type, COUNT(*) AS count
FROM metadata
GROUP BY cell_type
ORDER BY count DESC
""").show()

from pyspark.sql.functions import isnull

# Filter out where 's3_image_path' is null
missing_image_paths_df = df_metadata_spark.filter(isnull("s3_image_path"))

# Count missing 's3_image_path'
missing_count = missing_image_paths_df.count()

print(f"Number of records with missing 's3_image_path': {missing_count}")

# Plotting with matplotlib after collecting data
class_counts = spark.sql("SELECT cell_type, COUNT(*) AS count FROM metadata GROUP BY cell_type").toPandas()
class_counts.set_index('cell_type').plot(kind='bar', title='Class Distribution')
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col

# filter out where 's3_image_path' is null
df_cleaned = df_metadata_spark.filter(col("s3_image_path").isNotNull())

# Use df_cleaned
df_cleaned.createOrReplaceTempView("clean_metadata")
print(f"Cleaned dataset count: {df_cleaned.count()}")


# Check again for nulls after cleaning
null_checks_after_cleaning = df_cleaned.select([count(when(col(c).isNull(), c)).alias(c) for c in df_cleaned.columns])
null_checks_after_cleaning.show()

# Logging for monitoring
if null_checks_after_cleaning.first().asDict().get("s3_image_path", 0) > 0:
    print("Unexpected nulls found in s3_image_path after cleaning")


# COMMAND ----------

spark.sql("""
SELECT cell_type, COUNT(*) AS total_cases, AVG(age) AS average_age
FROM metadata
WHERE age > 30 AND sex = 'male'
GROUP BY cell_type
ORDER BY total_cases DESC
""").show()

# View specific patterns in the data
spark.sql("""
SELECT localization, cell_type, COUNT(*) AS count
FROM metadata
WHERE sex = 'female'
GROUP BY localization, cell_type
ORDER BY count DESC
LIMIT 10
""").show()

# View statistics across different categories.
spark.sql("""
SELECT cell_type, MAX(age) as max_age, MIN(age) as min_age, AVG(age) as avg_age
FROM metadata
WHERE localization IN ('face', 'back', 'chest')
GROUP BY cell_type
""").show()

# ID specific conditions that might be prevalent within certain demographic groups.
spark.sql("""
SELECT cell_type, SUM(CASE WHEN age > 50 THEN 1 ELSE 0 END) AS older_than_50
FROM metadata
GROUP BY cell_type
""").show()

# View all counts for each class to determine balance.
spark.sql("""
SELECT cell_type, COUNT(*) AS total_count
FROM metadata
GROUP BY cell_type
ORDER BY total_count DESC;
""").show()

# COMMAND ----------

from pyspark.sql.functions import col, rand, lit, explode, array_repeat

# Load the classes and their counts
class_counts_df = df_metadata_spark.groupBy("cell_type").count()

# Initialize an empty DataFrame to collect balanced data
balanced_df = None

for row in class_counts_df.collect():
    class_name = row['cell_type']
    count = row['count']
    class_df = df_metadata_spark.filter(col("cell_type") == class_name)

    if count > 600:
        # Undersample larger classes to 600
        sampled_df = class_df.sample(withReplacement=False, fraction=600 / count, seed=42)
    elif count < 600:
        # Oversample smaller classes to approximately 600
        oversample_factor = (600 // count) + 1
        sampled_df = class_df.withColumn("dummy", explode(array_repeat(lit(1), oversample_factor)))
        sampled_df = sampled_df.drop("dummy").limit(600)
    else:
        sampled_df = class_df  # Use as is if if count is exactly 600

    # Combine
    balanced_df = sampled_df if balanced_df is None else balanced_df.union(sampled_df)

# Shuffle
balanced_df = balanced_df.orderBy(rand())

# Filter out records where 's3_image_path' is null
balanced_df = balanced_df.filter(col("s3_image_path").isNotNull())

# Show the balanced data's distribution and filtered count
balanced_df.groupBy("cell_type").count().show()
print("Filtered DataFrame count:", balanced_df.count())

# COMMAND ----------

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to fetch and process images
def fetch_and_process_image(path):
    if path is None:
        print("No path provided")
        return np.zeros((128, 128, 3))  # Return a zero array if path is None
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=path)
        img = Image.open(io.BytesIO(obj['Body'].read()))
        img = img.resize((128, 128))
        img = img.convert('RGB')
        return np.array(img).astype('float32') / 255.0
    except Exception as e:
        print(f"Failed to load image {path}: {str(e)}")
        return np.zeros((128, 128, 3))

def process_images(paths):
    with ThreadPoolExecutor(max_workers=10) as executor:
        images = list(executor.map(fetch_and_process_image, paths))
    return np.array(images)

# Convert Spark DataFrame to Pandas DataFrame for image processing
pandas_balanced_df = balanced_df.toPandas()

# Apply image fetching and processing
images = process_images(pandas_balanced_df['s3_image_path'].tolist())
labels = pandas_balanced_df['cell_type_idx'].values

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training the model with augmentation on the training dataset
train_data_gen = data_gen.flow(X_train, y_train, batch_size=32)
model.fit(
    train_data_gen,
    epochs=30,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // 32,
    validation_steps=len(X_val) // 32
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")


# COMMAND ----------

train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
