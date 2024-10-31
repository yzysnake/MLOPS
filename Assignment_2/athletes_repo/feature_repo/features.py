from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Float32, String, Int64

# Define a project for the feature repo
project = Project(name="athletes_project", description="A project for athletes data")

# Define the entity
athlete = Entity(
    name="athlete_id",
    join_keys=["athlete_id"],
    description="Unique identifier for athletes",
)

# Define the predictor v2 data source
athletes_predictors_v1_source = FileSource(
    name='athletes_predictors_v1',
    path="data/athletes_predictors_df_v1.parquet",
    timestamp_field="event_timestamp")

# Define the predictor Feature 
athletes_predictors_features_v1 = FeatureView(
    name="athletes_predictors_feature_view_v1",
    entities=[athlete],
    ttl=timedelta(days=1),
    schema=[
        Field(name="gender", dtype=Int64),
        Field(name="age", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
    ],
    online=True,
    source=athletes_predictors_v1_source,
)

# Define the predictor v2 data source
athletes_predictors_v2_source = FileSource(
    name='athletes_predictors_v2',
    path="data/athletes_predictors_df_v2.parquet",
    timestamp_field="event_timestamp")

# Define the predictor Feature View 
athletes_predictors_features_v2 = FeatureView(
    name="athletes_predictors_feature_view_v2",
    entities=[athlete],
    ttl=timedelta(days=1),
    schema=[
        Field(name="gender", dtype=Int64),
        Field(name="age", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
        Field(name="experience_start_with_coach", dtype=Int64),
        Field(name="experience_have_certificate", dtype=Int64),
        Field(name="eat_on_diet", dtype=Int64)
    ],
    online=True,
    source=athletes_predictors_v2_source,
)

# Define the target data source
athletes_target_source = FileSource(
    name='athletes_target',
    path="data/athletes_target_df.parquet",
    timestamp_field="event_timestamp")

# Define the target Feature View with all columns
athletes_target_features = FeatureView(
    name="athletes_target_feature_view",
    entities=[athlete],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_lift", dtype=Float32),
        Field(name="candj", dtype=Float32),
        Field(name="snatch", dtype=Float32),
        Field(name="deadlift", dtype=Float32),
        Field(name="backsq", dtype=Float32)
    ],
    online=True,
    source=athletes_target_source,
)