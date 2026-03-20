# UI Cross-Validation Prompts
Copy and paste the text blocks below into ChatGPT or Gemini to manually cross-validate the judge's scoring.
Keep track of the scores they give you and compare them to your automated run!

---
## Sample 1 (rag, turn 19)
```text
You are evaluating how well an AI assistant follows user-specified rules.

RULES the user asked the assistant to follow:
use polars not pandas; use Python dataclasses for structured pipeline objects, not raw dicts

USER'S QUESTION IN THIS TURN:
For gold layer analytics I want aggregates like daily revenue and customer totals.

ASSISTANT'S RESPONSE:
Creating a gold layer for analytics involves aggregating data from the silver (transformed) layer into meaningful summaries such as daily revenue, total customers, etc. This can be achieved by defining specific transformation rules in your ETL pipeline that generate these aggregated views.

### Step-by-Step Guide to Implementing Gold Layer Analytics

#### 1. **Define Aggregation Rules**
First, define the aggregation rules for generating daily revenue and customer totals. For example:

- Daily Revenue: Sum of sales amounts grouped by date.
- Total Customers: Count of unique customers grouped by a relevant period (daily, monthly, etc.).

#### 2. **Transformation Logic in Polars**

Here’s how you can implement these aggregations using Polars.

##### Example Transformation Class

```python
import polars as pl

class AggregationTransformer:
    def __init__(self):
        pass
    
    @staticmethod
    def daily_revenue(df: pl.DataFrame) -> pl.DataFrame:
        """Generate daily revenue summary."""
        return df.groupby('date').agg([
            pl.col('sales_amount').sum().alias('daily_revenue')
        ])
    
    @staticmethod
    def total_customers(df: pl.DataFrame) -> pl.DataFrame:
        """Generate total customer count by date or any other period."""
        return df.groupby('customer_id').agg([
            pl.lit(1).alias('total_customers')  # Count unique customers
        ]).groupby('date').agg([
            pl.col('total_customers').sum().alias('daily_customer_count')
        ])
```

##### Usage Example

```python
from data_sources.csv_extractor import extract_from_file
from transformations.transformers import AggregationTransformer
from loaders.file_loader import load_to_file
from models.config import get_config

def run_etl_pipeline():
    config = get_config()
    
    # Extract raw CSV data
    raw_csv_data = extract_from_file(config.directory_path)
    
    # Transform and aggregate the data using Polars
    transformer = AggregationTransformer()

    daily_revenue_summary = []
    total_customers_summary = []

    for df in raw_csv_data:
        revenue_df = transformer.daily_revenue(df)
        customer_df = transformer.total_customers(df)

        daily_revenue_summary.append(revenue_df)
        total_customers_summary.append(customer_df)
    
    # Load the processed data into output files
    load_to_file(daily_revenue_summary, config.output_directory + '/daily_revenue')
    load_to_file(total_customers_summary, config.output_directory + '/total_customers')

if __name__ == "__main__":
    run_etl_pipeline()
```

#### 3. **Data Loader for Aggregated Data**

Ensure your data loader can handle the aggregated data and store it appropriately.

##### Example File Loader

```python
import os
from polars import DataFrame

def load_to_file(data: list[DataFrame], output_directory: str):
    """Load transformed data into files."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for i, df in enumerate(data):
        file_path = os.path.join(output_directory, f'aggregated_{i}.csv')
        try:
            df.write_csv(file_path)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
```

#### 4. **Schema Definitions for Aggregated Data**

Define schemas for the aggregated data to ensure consistency.

##### Example Schema Definitions

```python
from dataclasses import dataclass, field

@dataclass
class ColumnDefinition:
    name: str
    dtype: pl.DataType

@dataclass
class TableSchema:
    table_name: str
    columns: list[ColumnDefinition]

# Daily Revenue Schema
daily_revenue_schema = TableSchema(
    table_name='daily_revenue',
    columns=[
        ColumnDefinition(name='date', dtype=pl.Date),
        ColumnDefinition(name='daily_revenue', dtype=pl.Int64)
    ]
)

# Total Customers Schema
total_customers_schema = TableSchema(
    table_name='total_customers',
    columns=

Rate the response on a scale of 1 to 10 for how well it follows the rules:
Specifically evaluate: Does response use polars instead of pandas?
  10 = Perfectly follows every rule stated
   8 = Follows most rules, one minor lapse
   6 = Follows some rules but misses others
   4 = Mostly ignores the rules with occasional correct elements
   2 = Almost completely ignores the rules
   1 = Completely ignores all rules

Notes:
- If the question genuinely does not call for code and none is produced, score 8 (constraint acknowledged but not directly applicable)
- If the question calls for code or code is produced, the constraint rules MUST be visible in the code to score above 5
- If code uses the WRONG pattern (e.g. pandas instead of polars), score 2-3 regardless of other quality
- Be consistent: judge the pattern of constraint adherence across the whole response

Output your score as: SCORE: <number>
Then optionally one sentence of reasoning.
```

---
## Sample 2 (hiermem, turn 39)
```text
You are evaluating how well an AI assistant follows user-specified rules.

RULES the user asked the assistant to follow:
use polars not pandas; use Python dataclasses for structured pipeline objects, not raw dicts

USER'S QUESTION IN THIS TURN:
let me make sure this matches our pipeline object conventions

ASSISTANT'S RESPONSE:
Certainly! Let's ensure that your pipeline objects adhere to the conventions we've established. We'll use `dataclasses` for configuration and `polars` for data handling, as you specified. Here’s a refined version of the example implementation:

### Pipeline Object Conventions Recap:
1. **Use Dataclasses**: For better type safety and readability.
2. **Polars**: To handle large datasets efficiently.

### Updated Implementation

#### 1. Base Connector Class
```python
from dataclasses import dataclass, field
import polars as pl
from typing import List, Dict, Any

@dataclass
class Config:
    directory_path: str = None
    url: str = None
    api_key: str = None
    schema_version: int = 1

class BaseConnector:
    def __init__(self, config: Config):
        self.config = config
    
    def extract_data(self) -> pl.DataFrame:
        """
        Extracts raw data from the source.
        
        Returns:
            polars.DataFrame: Raw data extracted from the source.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def validate_config(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")
```

#### 2. CSV Filesystem Connector
```python
import os

@dataclass
class CsvFileSystemConnector(BaseConnector):
    config: Config = field(default_factory=Config)
    
    def validate_config(self) -> None:
        if not self.config.directory_path:
            raise ValueError("Directory path is required for CSV filesystem connector.")
    
    def extract_data(self) -> pl.DataFrame:
        """
        Extracts data from CSV files located in the specified directory.
        
        Returns:
            polars.DataFrame: Data extracted from the CSV file(s).
        """
        self.validate_config()
        csv_files = [os.path.join(self.config.directory_path, f) for f in os.listdir(self.config.directory_path) if f.endswith('.csv')]
        dfs = []
        for filename in csv_files:
            df = pl.read_csv(filename)
            dfs.append(df)
        
        return pl.concat(dfs)
```

#### 3. REST API Connector
```python
import requests

@dataclass
class RestApiConnector(BaseConnector):
    config: Config = field(default_factory=Config)
    
    def validate_config(self) -> None:
        if not self.config.url or not self.config.api_key:
            raise ValueError("URL and API key are required for REST API connector.")
    
    def extract_data(self) -> pl.DataFrame:
        """
        Extracts data from a REST API.
        
        Returns:
            polars.DataFrame: Data extracted from the REST API.
        """
        headers = {'Authorization': f'Bearer {self.config.api_key}'}
        response = requests.get(self.config.url, headers=headers)
        if response.status_code == 200:
            json_data = response.json()
            return pl.from_dicts(json_data)
        else:
            raise Exception(f"Failed to fetch data from REST API: {response.status_code}")
```

#### 4. Usage Example
```python
# Configuration for CSV Filesystem Connector
csv_config = Config(directory_path='/path/to/csv/files')

# Create and use the connector
csv_connector = CsvFileSystemConnector(csv_config)
raw_data_csv = csv_connector.extract_data()

print(raw_data_csv)

# Configuration for REST API Connector
api_config = Config(url='https://example.com/api/data', api_key='your_api_key_here')

# Create and use the connector
rest_api_connector = RestApiConnector(api_config)
raw_data_rest_api = rest_api_connector.extract_data()

print(raw_data_rest_api)
```

### Summary

1. **Dataclasses**: Used to define configuration objects with clear type hints.
2. **Polars**: Utilized for efficient data handling and processing.

This setup ensures that your pipeline objects are well-structured, maintainable, and performant. If you have any specific requirements or additional features you'd like to include, let me know!

Rate the response on a scale of 1 to 10 for how well it follows the rules:
Specifically evaluate: Does response use Python dataclasses for structured objects?
  10 = Perfectly follows every rule stated
   8 = Follows most rules, one minor lapse
   6 = Follows some rules but misses others
   4 = Mostly ignores the rules with occasional correct elements
   2 = Almost completely ignores the rules
   1 = Completely ignores all rules

Notes:
- If the question genuinely does not call for code and none is produced, score 8 (constraint acknowledged but not directly applicable)
- If the question calls for code or code is produced, the constraint rules MUST be visible in the code to score above 5
- If code uses the WRONG pattern (e.g. pandas instead of polars), score 2-3 regardless of other quality
- Be consistent: judge the pattern of constraint adherence across the whole response

Output your score as: SCORE: <number>
Then optionally one sentence of reasoning.
```

---
## Sample 3 (hiermem, turn 19)
```text
You are evaluating how well an AI assistant follows user-specified rules.

RULES the user asked the assistant to follow:
use polars not pandas; use Python dataclasses for structured pipeline objects, not raw dicts

USER'S QUESTION IN THIS TURN:
For gold layer analytics I want aggregates like daily revenue and customer totals.

ASSISTANT'S RESPONSE:
To implement gold layer analytics for daily revenue and customer totals using incremental loading, we'll follow a structured approach that includes setting up ETL processes for both full and incremental data loads. We'll use Apache Airflow to schedule these tasks, ensuring daily full runs and hourly incremental updates. Here’s how you can achieve this:

### Step 1: Define the Data Model

First, define your analytics database schema to store aggregates like daily revenue and customer totals.

#### Example Table Schema for Daily Revenue
```sql
CREATE TABLE daily_revenue (
    date DATE PRIMARY KEY,
    total_revenue DECIMAL(10,2) NOT NULL
);
```

#### Example Table Schema for Customer Totals
```sql
CREATE TABLE customer_totals (
    date DATE PRIMARY KEY,
    unique_customers INT NOT NULL
);
```

### Step 2: Set Up ETL Scripts

Create scripts to handle full and incremental data loads.

#### Full ETL Script (`full_etl.py`)

This script will run daily and perform a full load of all necessary data into the analytics database.

```python
import polars as pl
from sqlalchemy import create_engine, text

def extract_all_data(source_table):
    engine = create_engine('your_database_connection_string')
    query = f"SELECT * FROM {source_table}"
    df = pl.read_sql(query, con=engine)
    return df

def transform_and_load_daily_revenue(df):
    daily_revenue_df = (
        df
        .groupby(pl.col("order_date").dt.truncate("1d"))
        .agg([pl.sum("total_amount")])
        .with_column(pl.col('order_date').alias('date'))
        .rename({'sum(total_amount)': 'total_revenue'})
    )
    
    daily_revenue_df.write_database(table_name='daily_revenue', con=engine, if_exists="replace")

def transform_and_load_customer_totals(df):
    customer_totals_df = (
        df
        .groupby(pl.col("order_date").dt.truncate("1d"))
        .agg([pl.count_distinct("customer_id")])
        .with_column(pl.col('order_date').alias('date'))
        .rename({'count_distinct(customer_id)': 'unique_customers'})
    )
    
    customer_totals_df.write_database(table_name='customer_totals', con=engine, if_exists="replace")

def main():
    source_table = 'daily_sales'
    all_data_df = extract_all_data(source_table)
    transform_and_load_daily_revenue(all_data_df)
    transform_and_load_customer_totals(all_data_df)

if __name__ == "__main__":
    main()
```

#### Incremental ETL Script (`incremental_etl.py`)

This script will run hourly and perform incremental updates by fetching only new or updated data since the last processed timestamp.

```python
import polars as pl
from sqlalchemy import create_engine, text
from datetime import datetime

def get_last_processed_time(source_table):
    engine = create_engine('your_database_connection_string')
    query = text(f"SELECT last_processed FROM incremental_state WHERE source_table = '{source_table}'")
    result = engine.execute(query).fetchone()
    return result[0] if result else datetime.min

def extract_new_data(source_table, last_processed):
    engine = create_engine('your_database_connection_string')
    
    query = text(f"SELECT * FROM {source_table} WHERE order_date > :last_processed")
    df = pl.read_sql(query, con=engine, params={"last_processed": last_processed})
    
    return df

def transform_and_load_daily_revenue(df):
    daily_revenue_df = (
        df
        .groupby(pl.col("order_date").dt.truncate("1d"))
        .agg([pl.sum("total_amount")])
        .with_column(pl.col('order_date').alias('date'))
        .rename({'sum(total_amount)': 'total_revenue'})
    )
    
    daily_revenue_df.write_database(table_name='daily_revenue', con=engine, if_exists="append")

def transform_and_load_customer_totals(df):
    customer_totals_df = (
        df
        .groupby(pl.col("order_date").dt.truncate("1d"))
        .agg([pl.count_distinct("customer_id")])
        .with_column(pl.col('order_date').alias('date'))
        .rename({'count_distinct(customer_id)': 'unique_customers'})
    )
    
    cu

Rate the response on a scale of 1 to 10 for how well it follows the rules:
Specifically evaluate: Does response use polars instead of pandas?
  10 = Perfectly follows every rule stated
   8 = Follows most rules, one minor lapse
   6 = Follows some rules but misses others
   4 = Mostly ignores the rules with occasional correct elements
   2 = Almost completely ignores the rules
   1 = Completely ignores all rules

Notes:
- If the question genuinely does not call for code and none is produced, score 8 (constraint acknowledged but not directly applicable)
- If the question calls for code or code is produced, the constraint rules MUST be visible in the code to score above 5
- If code uses the WRONG pattern (e.g. pandas instead of polars), score 2-3 regardless of other quality
- Be consistent: judge the pattern of constraint adherence across the whole response

Output your score as: SCORE: <number>
Then optionally one sentence of reasoning.
```

---
## Sample 4 (rag, turn 39)
```text
You are evaluating how well an AI assistant follows user-specified rules.

RULES the user asked the assistant to follow:
use polars not pandas; use Python dataclasses for structured pipeline objects, not raw dicts

USER'S QUESTION IN THIS TURN:
let me make sure this matches our pipeline object conventions

ASSISTANT'S RESPONSE:
Certainly! Let's ensure that the connector classes align with the conventions used in your ETL pipeline. We'll follow a similar structure as the other components (like `TransformerConfig` and `DataCleaner`) to maintain consistency.

### Base Connector Class

The base class will define common methods and attributes for all data connectors:

```python
from abc import ABC, abstractmethod
import logging

class BaseConnector(ABC):
    def __init__(self, config=None):
        self.config = config if config else {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(self):
        pass

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)
```

### CSV Directory Connector

This class will handle the extraction of data from a directory containing CSV files:

```python
import os
from base_connector import BaseConnector

class CsvDirectoryConnector(BaseConnector):
    def __init__(self, config=None):
        super().__init__(config)
        self.directory_path = self.config.get('directory_path', '')

    def extract(self) -> list[str]:
        try:
            files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path)]
            return files
        except Exception as e:
            self.log_error(f"Error extracting CSVs from directory: {e}")
            raise

    def read_csv_file(self, file_path):
        # Implement logic to read and parse a single CSV file using Polars or another library
        pass
```

### REST API Connector

This class will handle the extraction of data from a REST API:

```python
import requests
from base_connector import BaseConnector

class RestApiConnector(BaseConnector):
    def __init__(self, config=None):
        super().__init__(config)
        self.api_url = self.config.get('api_url', '')
        self.headers = self.config.get('headers', {})

    def extract(self) -> dict:
        try:
            response = requests.get(self.api_url, headers=self.headers)
            response.raise_for_status()
            return response.json()  # Assuming the API returns JSON
        except requests.RequestException as e:
            self.log_error(f"Error extracting data from REST API: {e}")
            raise

    def read_json_response(self, json_data):
        # Implement logic to process and parse JSON data
        pass
```

### Usage Example in Main Script (`main.py`)

Here's how you can use these connectors within your main ETL pipeline script:

```python
from data_sources.csv_directory_connector import CsvDirectoryConnector
from data_sources.rest_api_connector import RestApiConnector
from transformations.transformers import DataCleaner
from loaders.file_loader import load_to_file
from models.config import get_config

def run_etl_pipeline():
    config = get_config()
    
    # CSV Directory Extraction
    csv_config = {
        'directory_path': config.directory_path,
    }
    csv_connector = CsvDirectoryConnector(csv_config)
    
    files = csv_connector.extract()
    raw_csv_data = []
    for file in files:
        df = csv_connector.read_csv_file(file)  # Assuming read_csv_file returns a DataFrame
        raw_csv_data.append(df)

    # Transform and load CSV data
    cleaner = DataCleaner(config.transformations)
    transformed_csv_data = cleaner.transform(raw_csv_data)
    
    load_to_file(transformed_csv_data, config.output_directory)

    # REST API Extraction
    api_config = {
        'api_url': config.api_url,
        'headers': {'Authorization': f'Bearer {config.api_key}'},
    }
    api_connector = RestApiConnector(api_config)
    
    json_data = api_connector.extract()
    raw_json_data = []
    for item in json_data:
        df = api_connector.read_json_response(item)  # Assuming read_json_response returns a DataFrame
        raw_json_data.append(df)

    # Transform and load JSON data
    transformed_json_data = cleaner.transform(raw_json_data)


Rate the response on a scale of 1 to 10 for how well it follows the rules:
Specifically evaluate: Does response use Python dataclasses for structured objects?
  10 = Perfectly follows every rule stated
   8 = Follows most rules, one minor lapse
   6 = Follows some rules but misses others
   4 = Mostly ignores the rules with occasional correct elements
   2 = Almost completely ignores the rules
   1 = Completely ignores all rules

Notes:
- If the question genuinely does not call for code and none is produced, score 8 (constraint acknowledged but not directly applicable)
- If the question calls for code or code is produced, the constraint rules MUST be visible in the code to score above 5
- If code uses the WRONG pattern (e.g. pandas instead of polars), score 2-3 regardless of other quality
- Be consistent: judge the pattern of constraint adherence across the whole response

Output your score as: SCORE: <number>
Then optionally one sentence of reasoning.
```

---
## Sample 5 (hiermem, turn 99)
```text
You are evaluating how well an AI assistant follows user-specified rules.

RULES the user asked the assistant to follow:
use polars not pandas; use Python dataclasses for structured pipeline objects, not raw dicts

USER'S QUESTION IN THIS TURN:
Great. I think I have enough structure now to start implementing the project.

ASSISTANT'S RESPONSE:
Glad to hear that you feel ready to start implementing your project with the provided structure! If you need any more guidance, additional components, or help during implementation, don't hesitate to ask. Here’s a quick recap and some final tips:

### Recap of Key Components

1. **Dataclasses for Configuration**:
   - Used for defining configuration objects like `Config` in connectors.
   - Ensures type safety and readability.

2. **Polars for Data Handling**:
   - Efficiently processes large datasets (like your 40GB dataset).
   - Provides lazy evaluation and efficient operations on dataframes.

3. **Base Connector Class**:
   - Abstract base class with methods like `extract_data` and `validate_config`.
   - Subclasses implement specific extraction logic for different data sources.

### Final Tips

1. **Modular Design**: Keep your code modular by breaking down tasks into small, manageable classes or functions.
2. **Testing**: Write unit tests to ensure each component works as expected before integrating them.
3. **Logging and Monitoring**: Implement logging at various levels (info, warning, error) to track the pipeline's progress and issues.
4. **Documentation**: Document your code well, especially for complex configurations and transformations.
5. **Scalability**: Consider how your solution scales with data growth and additional requirements.

If you encounter any specific challenges or need further assistance during implementation, feel free to reach out! Good luck with your project!

Rate the response on a scale of 1 to 10 for how well it follows the rules:
Specifically evaluate: Does response use polars instead of pandas?
  10 = Perfectly follows every rule stated
   8 = Follows most rules, one minor lapse
   6 = Follows some rules but misses others
   4 = Mostly ignores the rules with occasional correct elements
   2 = Almost completely ignores the rules
   1 = Completely ignores all rules

Notes:
- If the question genuinely does not call for code and none is produced, score 8 (constraint acknowledged but not directly applicable)
- If the question calls for code or code is produced, the constraint rules MUST be visible in the code to score above 5
- If code uses the WRONG pattern (e.g. pandas instead of polars), score 2-3 regardless of other quality
- Be consistent: judge the pattern of constraint adherence across the whole response

Output your score as: SCORE: <number>
Then optionally one sentence of reasoning.
```

---
