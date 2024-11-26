from time import sleep

from pyspark.sql import SparkSession
from transformers import pipeline
from pyspark.sql.functions import from_json, col, when, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from config.config import config

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def sentiment_analysis(comment) -> str:
    if comment:
        result = sentiment_analyzer(comment)
        if result:
            # Extract sentiment (e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL')
            return result[0]['label'].upper()
    return "Empty"

def start_streaming(spark):
    topic = 'customers_review'
    while True:
        try:
            stream_df = (spark.readStream.format("socket")
                         .option("host", "0.0.0.0")
                         .option("port", 9999)
                         .load()
                         )

            schema = StructType([
                StructField("review_id", StringType()),
                StructField("user_id", StringType()),
                StructField("business_id", StringType()),
                StructField("stars", FloatType()),
                StructField("date", StringType()),
                StructField("text", StringType()),
                StructField("feedback", StringType())
            ])

            stream_df = stream_df.select(from_json(col('value'), schema).alias("data")).select(("data.*"))

            sentiment_analysis_udf = udf(sentiment_analysis, StringType())

            stream_df = stream_df.withColumn('feedback',
                                             when(col('text').isNotNull(), sentiment_analysis_udf(col('text')))
                                             .otherwise(None)
                                             )

            kafka_df = stream_df.selectExpr("CAST(review_id AS STRING) AS key", "to_json(struct(*)) AS value")

            query = (kafka_df.writeStream
                     .format("kafka")
                     .option("kafka.bootstrap.servers", config['kafka']['bootstrap.servers'])
                     .option("kafka.security.protocol", config['kafka']['security.protocol'])
                     .option('kafka.sasl.mechanism', config['kafka']['sasl.mechanisms'])
                     .option('kafka.sasl.jaas.config',
                             'org.apache.kafka.common.security.plain.PlainLoginModule required username="{username}" '
                             'password="{password}";'.format(
                                 username=config['kafka']['sasl.username'],
                                 password=config['kafka']['sasl.password']
                             ))
                     .option('checkpointLocation', '/tmp/checkpoint')
                     .option('topic', topic)
                     .start()
                     .awaitTermination()
                     )

        except Exception as e:
            print(f'Exception encountered: {e}. Retrying in 10 seconds')
            sleep(10)

if __name__ == "__main__":
    spark_conn = SparkSession.builder.appName("SocketStreamConsumer").getOrCreate()
    start_streaming(spark_conn)
