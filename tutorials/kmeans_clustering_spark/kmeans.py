from operator import add

import flytekit
from flytekit import Resources, task, workflow
import flytekit.deck
from flytekitplugins.spark import Databricks, Spark
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import plotly.graph_objects as go
import numpy as np
import csv
from flytekit.types.file import CSVFile
import tempfile
from flytekit.image_spec import ImageSpec


image_spec = ImageSpec(
    builder="envd",
    name="spark",
    registry="ghcr.io/unionai-oss",
    requirements="requirements.txt")


@task(
    task_config=Spark(
        spark_conf={
            "spark.driver.memory": "1000M",
            "spark.executor.memory": "1000M",
            "spark.executor.cores": "1",
            "spark.executor.instances": "2",
            "spark.driver.cores": "1",
            "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar",
        },
        executor_path="/usr/bin/python3",
        applications_path="local:///usr/local/bin/entrypoint.py",
    ),
    # task_config=Databricks(
    #     spark_conf={
    #         "spark.driver.memory": "1000M",
    #         "spark.executor.memory": "1000M",
    #         "spark.executor.cores": "1",
    #         "spark.executor.instances": "2",
    #         "spark.driver.cores": "1",
    #     },
    #     databricks_conf={
    #         "run_name": "flytekit databricks plugin example",
    #         "new_cluster": {
    #             "spark_version": "11.0.x-scala2.12",
    #             "node_type_id": "r3.xlarge",
    #             "aws_attributes": {
    #                 "availability": "ON_DEMAND",
    #                 "instance_profile_arn": "arn:aws:iam::<AWS_ACCOUNT_ID_DATABRICKS>:instance-profile/databricks-flyte-integration",
    #             },
    #             "num_workers": 4,
    #         },
    #         "timeout_seconds": 3600,
    #         "max_retries": 1,
    #     },
    # ),
    limits=Resources(mem="2000M"),
    enable_deck=True,
    container_image=image_spec,
    # Uncomment the following line to enable caching
    # cache=True,
    # cache_version="1",
)
def kmeans_cluster(dataset_file: CSVFile) -> int:
    sess = flytekit.current_context().spark_session
    dataset = sess.read.csv(dataset_file.remote_source, header=True, inferSchema=True) 

    # Assembles columns 'x', 'y', and 'z' into a single vector column 'features'
    assembler = VectorAssembler(inputCols=["x", "y", "z"], outputCol="features")
    dataset = assembler.transform(dataset)

    # Trains a k-means model.
    kmeans = KMeans().setK(2).setSeed(1)
    model = kmeans.fit(dataset)

    # Make predictions
    predictions = model.transform(dataset)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)
    # $example off$

    # Collect data from Spark DataFrame
    data_collected = predictions.select("features", "prediction").collect()

    # Extract x, y, z values and predictions
    x_values = [row["features"][0] for row in data_collected]
    y_values = [row["features"][1] for row in data_collected]
    z_values = [row["features"][2] for row in data_collected]
    predictions = [row["prediction"] for row in data_collected]

    # Plot
    fig = go.Figure()

    # Add points
    fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values,
                            mode='markers',
                            marker=dict(size=5,
                                        color=predictions,  # set color to cluster labels
                                        colorscale='Viridis',   # choose a colorscale
                                        opacity=0.8),
                            name='Data Points'))

    # Extract cluster centers
    centers = model.clusterCenters()
    centers_x = [center[0] for center in centers]
    centers_y = [center[1] for center in centers]
    centers_z = [center[2] for center in centers]

    # Add cluster centers to the plot
    fig.add_trace(go.Scatter3d(x=centers_x, y=centers_y, z=centers_z,
                            mode='markers',
                            marker=dict(size=10,
                                        color='Red',
                                        opacity=0.3),  # cluster centers in red
                            name='Cluster Centers'))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Adjust layout for full page
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, t=0))

    flytekit.Deck("Visualizer", fig.to_html(full_html=False, include_plotlyjs='cdn'))
    return len(centers)


@task(enable_deck=True, 
      cache=True, cache_version="1",
      container_image=image_spec)
def generate_data(dataset_size: int) -> CSVFile:
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate 50 points around (1, 1, 1) with a smaller spread
    cluster_1 = np.random.normal(loc=1, scale=0.2, size=(int(dataset_size/2), 3))

    # Generate 50 points around (3, 3, 3) with a smaller spread
    cluster_2 = np.random.normal(loc=3, scale=0.2, size=(int(dataset_size/2), 3))

    # Combine the clusters to form the dataset
    coordinates = np.vstack((cluster_1, cluster_2))

    # Create a temporary file to store the CSV data
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    try:
        writer = csv.writer(temp_file)
        # Write the header
        writer.writerow(['x', 'y', 'z'])
        # Write the coordinates
        for row in coordinates:
            writer.writerow(row)
    finally:
        temp_file.close()

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
                               mode='markers',
                               marker=dict(size=5,
                                           color=np.concatenate([np.zeros(50), np.ones(50)]),  # Example cluster labels
                                           colorscale='Viridis',
                                           opacity=0.8),
                               name='Data Points'))
    
    # Adjust layout for full page
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, t=0))

    flytekit.Deck("Visualizer", fig.to_html(full_html=False, include_plotlyjs='cdn'))

    return CSVFile(path=temp_file.name)


@workflow
def kmeans(dataset_size: int=100) -> int:
    dataset = generate_data(dataset_size=dataset_size)
    clusters = kmeans_cluster(dataset_file=dataset)
    return clusters

