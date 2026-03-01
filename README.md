Lab 3: Data Preprocessing in Azure Databricks

1. Introduction
This lab demonstrates the implementation of a data preprocessing pipeline using Azure Databricks and Apache Spark within a Lakehouse (medallion) architecture. The objective is to transform raw and processed Amazon Electronics review data into a curated, analytics-ready dataset. The pipeline focuses on data cleaning, enrichment with metadata, and preparation of a Gold dataset suitable for visualization and machine learning applications.

2. ETL Pipeline Design
The data preprocessing workflow is divided into three Databricks notebooks, each responsible for a specific stage of the pipeline. The process begins with loading and cleaning review data, followed by enrichment with product metadata, and concludes with the creation of a curated Gold dataset. This modular design ensures clarity, reusability, and ease of orchestration.

3. Data Enrichment
After cleaning the review data, additional contextual information is incorporated through enrichment. This includes joining review records with product metadata such as title, brand, and price. Further enrichment opportunities could include generating derived features (e.g., review length), sentiment analysis of review text, aggregated metrics such as average ratings per product or brand, and time-based features to support trend analysis. These enhancements increase the analytical value of the dataset.

4. Data Analysis and Visualization
The curated Gold dataset was loaded into a separate Databricks notebook for exploratory analysis and visualization. This dataset represents a high-quality, structured version of the reviews data.

4.1 Average Rating Over Time
The first visualization illustrates how average product ratings change over time based on the review year. The trend shows variations in customer satisfaction across different periods, highlighting phases of decline and recovery. This analysis helps identify long-term patterns in customer feedback and potential shifts in product quality or user expectations.

4.2 Rating Distribution
The second visualization presents the distribution of review ratings from one to five stars. The results indicate that higher ratings dominate the dataset, with five-star and four-star reviews occurring most frequently. This suggests overall positive customer sentiment toward the products included in the dataset.

5. Technologies Used
Apache Spark
Apache Spark is a distributed data processing engine designed for large-scale analytics. It was used to load data from Azure Data Lake, perform transformations, join datasets, and write processed outputs efficiently using Spark DataFrames.

Azure Databricks
Azure Databricks is a cloud-based analytics platform built on Apache Spark. It provides a managed environment for developing and executing Spark workloads. In this lab, Databricks was used to run notebooks, manage compute resources, and orchestrate the pipeline using Databricks Jobs.

Parquet
Parquet is a columnar storage format optimized for analytical workloads. By storing data in a column-oriented structure, Parquet improves query performance and reduces storage size. It was used for storing both the Silver and Gold datasets.

Delta Lake
Delta Lake is a storage layer built on top of Parquet that adds reliability features such as ACID transactions and data versioning. Although Delta tables were not explicitly implemented in this lab, Delta Lake is commonly used in Databricks environments to ensure data consistency in production pipelines.

6. Pipeline Implementation Overview

The pipeline begins by loading review and metadata files from Azure Data Lake Storage using the abfss:// protocol. Spark DataFrames are created by reading Parquet and JSON files, enabling scalable processing.

Data cleaning steps ensure quality by filtering invalid records, enforcing valid rating ranges, and validating review text. The cleaned data is then enriched through a left join with product metadata using the product identifier.

Following enrichment, the processed data is written to the Silver layer. A curated Gold dataset is created by selecting a final set of features and storing the output in Parquet format. The entire workflow is orchestrated using Databricks Jobs to ensure correct execution order and data dependencies.

7. Conclusion
This lab showcases a practical implementation of a Lakehouse-style data preprocessing pipeline using Azure Databricks and Apache Spark. Through structured ETL steps and efficient storage formats, the pipeline produces a reliable and analytics-ready Gold dataset that can be used for reporting, visualization, and machine learning tasks.


Q1.How do the notebooks map to the ETL process?

The three Databricks notebooks directly map to the ETL (Extract, Transform, Load) process.
The first notebook (01_load_and_clean_reviews) is responsible for extracting the processed reviews data from Azure Data Lake and performing data cleaning and validation, which is part of the transformation stage.
The second notebook (02_enrich_with_metadata) further transforms the data by enriching the cleaned reviews with product metadata through a join operation, producing a refined Silver dataset.
The third notebook (03_write_gold_features_v1) loads the enriched Silver data and selects the final set of features, writing a curated Gold dataset that is ready for analytics and machine learning.

Q2.What other enrichment can you do to the current Gold layer?

Other enrichments that can be applied to the Gold layer include adding derived features such as the length of the review text, sentiment scores extracted from the review text, or aggregated metrics like the average rating per product or per brand. Time-based features such as review month or season can also be added to support trend analysis. These enrichments make the dataset more useful for analytics and machine learning tasks.

<img width="975" height="585" alt="image" src="https://github.com/user-attachments/assets/a4be0548-8717-4149-82e2-b2b6ef488c86" />
<img width="975" height="433" alt="image" src="https://github.com/user-attachments/assets/e92e9931-df89-47c1-88cc-53e53a8d4a3a" />
Explanation:
This visualization shows how the average product rating changes over time based on the review year. The line chart indicates that ratings were relatively high in earlier years, then declined around the mid-2000s, before gradually increasing again in later years. This trend suggests changes in customer satisfaction over time and helps identify periods where product quality or customer expectations may have shifted.

<img width="975" height="423" alt="image" src="https://github.com/user-attachments/assets/1ae69d8b-a9cd-4a36-a162-0748600536bc" />
<img width="975" height="609" alt="image" src="https://github.com/user-attachments/assets/4b6cae8c-52c2-4a91-97b8-3f687dafae6d" />
Explanation:
This visualization displays the distribution of review ratings from 1 to 5 stars. The chart shows that 5-star ratings are the most frequent, followed by 4-star ratings, while lower ratings occur much less often. This indicates that the majority of customers provided positive feedback, suggesting overall high customer satisfaction with the products in the dataset.


**Screenshots**

<img width="975" height="372" alt="image" src="https://github.com/user-attachments/assets/20230820-310b-46cc-a0cc-2102deb7a56f" />
<img width="975" height="395" alt="image" src="https://github.com/user-attachments/assets/de555ba6-565b-4527-8fc6-10c38502fe9e" />
<img width="975" height="396" alt="image" src="https://github.com/user-attachments/assets/39d6c7f8-20c5-4bcf-9f8e-8739f7de1c43" />
<img width="975" height="294" alt="image" src="https://github.com/user-attachments/assets/5a1735d4-cce6-4bf7-bd40-06ba19fc9032" />
<img width="975" height="372" alt="image" src="https://github.com/user-attachments/assets/a35ce220-104e-4729-99e7-7c578b96379a" />
<img width="975" height="410" alt="image" src="https://github.com/user-attachments/assets/d96df670-565e-4eba-a6cd-5393ecf672ae" />
<img width="975" height="567" alt="image" src="https://github.com/user-attachments/assets/c27a7e66-264a-437a-9639-943b1e7dd146" />
<img width="975" height="556" alt="image" src="https://github.com/user-attachments/assets/22c05e29-8950-491b-a953-b36c95781753" />
<img width="975" height="556" alt="image" src="https://github.com/user-attachments/assets/55081e32-ce0c-4d5b-96bc-8839e34a96ad" />
<img width="975" height="249" alt="image" src="https://github.com/user-attachments/assets/83de3164-efc8-4e23-8cad-3c56eb7355de" />
<img width="975" height="426" alt="image" src="https://github.com/user-attachments/assets/37f7b7c0-8663-4bd6-893e-0f3e530f56a5" />
<img width="975" height="414" alt="image" src="https://github.com/user-attachments/assets/750f1f2f-16e3-4cfe-bb0d-a63c5e1d12c0" />
<img width="975" height="331" alt="image" src="https://github.com/user-attachments/assets/ba06e410-bd6a-4e21-8a76-0da26318653a" />
<img width="975" height="413" alt="image" src="https://github.com/user-attachments/assets/e125f016-eeda-4f6d-8e9f-43b9bebe3288" />
<img width="975" height="423" alt="image" src="https://github.com/user-attachments/assets/b2f9eda1-5a4c-48b2-abe5-c776f1a1129a" />


