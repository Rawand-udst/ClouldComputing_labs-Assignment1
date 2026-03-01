# DSAI3202---Lab2--Data-Ingestion-in-Azure

In this lab, I worked with an existing Azure environment to build a complete data ingestion pipeline for the Amazon Electronics dataset. I used Azure Storage as a data lake and organized the data into raw, processed, and curated containers to clearly separate unprocessed and processed data.

I ingested two datasets into the raw container. The electronics reviews dataset was downloaded and uploaded programmatically using an Azure ML Compute Instance and AzCopy, while the product metadata file was uploaded through the Azure Portal. During ingestion, I identified that the metadata file was not valid JSON, so I fixed it by converting each line from a Python dictionary into valid line-delimited JSON and re-uploaded the corrected version. This ensured the data could be reliably used in downstream processing.

Using Azure Data Factory, I created datasets for the raw reviews data and the processed output. I then built a Mapping Data Flow that reads the raw JSON reviews, derives a review_year column from the Unix timestamp, and writes the output in Parquet format partitioned by year. A pipeline was created to run this data flow, and the results were verified in the processed container.

To demonstrate automation, I added a scheduled trigger that runs the pipeline automatically on a daily basis. All changes were published in Azure Data Factory to activate the pipeline and trigger.

Throughout the lab, I maintained a clear GitHub workflow by committing changes with meaningful messages and using branches to separate development work from the stable version on main. I also practiced proper Azure resource management by using compute resources only when needed and avoiding unnecessary running services.

This lab helped me understand how data ingestion, transformation, automation, version control, and cloud resource management work together in a real-world data engineering workflow.

**Screenshot**
<img width="975" height="566" alt="image" src="https://github.com/user-attachments/assets/00bbaa71-921a-420a-b970-e46f6e954437" />

<img width="975" height="380" alt="image" src="https://github.com/user-attachments/assets/3c85de2c-5217-47e3-b2e7-9fff26cde5e2" />

<img width="975" height="171" alt="image" src="https://github.com/user-attachments/assets/09908af9-7619-44f9-a4d5-1557b3f3f4ae" />

<img width="975" height="342" alt="image" src="https://github.com/user-attachments/assets/58877329-d902-4a10-be66-02213cb59e2c" />

<img width="975" height="395" alt="image" src="https://github.com/user-attachments/assets/d6ee0fba-53c1-404b-90d7-8312b351baec" />

<img width="975" height="609" alt="image" src="https://github.com/user-attachments/assets/baa02096-bef5-4641-84a1-e9af8d5760b9" />

<img width="975" height="612" alt="image" src="https://github.com/user-attachments/assets/da548c54-73f8-4160-b716-540f37574f17" />

<img width="975" height="427" alt="image" src="https://github.com/user-attachments/assets/e4d9f3af-e038-4e94-9a29-c518b9cb8bca" />



