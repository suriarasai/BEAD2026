# Databricks notebook source
# MAGIC %md
# MAGIC # Taxi Fleet Exploratory Data Analysis
# MAGIC
# MAGIC **Dataset Overview:**
# MAGIC - Drivers.csv: Information about taxi drivers
# MAGIC - Taxi.csv: Information about taxi vehicles
# MAGIC - TaxiType.csv: Different types of taxis and their pricing
# MAGIC
# MAGIC **Objective:** Perform comprehensive exploratory data analysis to understand the taxi fleet operations

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Data Loading

# COMMAND ----------

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload the CSV files to DBFS
# MAGIC
# MAGIC Before running the next cells, upload your CSV files to Databricks:
# MAGIC 1. Click on "Data" in the left sidebar
# MAGIC 2. Click "Create Table" 
# MAGIC 3. Upload: Drivers.csv, Taxi.csv, TaxiType.csv
# MAGIC 4. Or use the file upload option in the notebook
# MAGIC
# MAGIC **Note:** Update the file paths below based on where you uploaded the files

# COMMAND ----------

# Define file paths (update these based on your DBFS location)
drivers_path = "/Volumes/workspace/rebu/raw/Drivers.csv"
taxi_path = "/Volumes/workspace/rebu/raw/Taxi.csv"
taxitype_path = "/Volumes/workspace/rebu/raw/TaxiType.csv"


# COMMAND ----------

# Load data into Spark DataFrames
drivers_df = spark.read.csv(drivers_path, header=True, inferSchema=True)
taxi_df = spark.read.csv(taxi_path, header=True, inferSchema=True)
taxitype_df = spark.read.csv(taxitype_path, header=True, inferSchema=True)

print("✓ Data loaded successfully!")
print(f"Drivers: {drivers_df.count()} records")
print(f"Taxis: {taxi_df.count()} records")
print(f"Taxi Types: {taxitype_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Schema and Structure

# COMMAND ----------

# Display schema for each dataset
print("=" * 60)
print("DRIVERS SCHEMA")
print("=" * 60)
drivers_df.printSchema()

print("\n" + "=" * 60)
print("TAXI SCHEMA")
print("=" * 60)
taxi_df.printSchema()

print("\n" + "=" * 60)
print("TAXI TYPE SCHEMA")
print("=" * 60)
taxitype_df.printSchema()

# COMMAND ----------

# Display sample data
print("DRIVERS - First 5 Records:")
display(drivers_df.limit(5))

# COMMAND ----------

print("TAXI - First 5 Records:")
display(taxi_df.limit(5))

# COMMAND ----------

print("TAXI TYPES - All Records:")
display(taxitype_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Assessment

# COMMAND ----------

# Check for null values in Drivers
print("NULL VALUES IN DRIVERS DATASET:")
print("-" * 60)
drivers_nulls = drivers_df.select([count(when(col(c).isNull(), c)).alias(c) for c in drivers_df.columns])
display(drivers_nulls)

# COMMAND ----------

# Check for null values in Taxi
print("NULL VALUES IN TAXI DATASET:")
print("-" * 60)
taxi_nulls = taxi_df.select([count(when(col(c).isNull(), c)).alias(c) for c in taxi_df.columns])
display(taxi_nulls)

# COMMAND ----------

# Check for null values in TaxiType
print("NULL VALUES IN TAXI TYPE DATASET:")
print("-" * 60)
taxitype_nulls = taxitype_df.select([count(when(col(c).isNull(), c)).alias(c) for c in taxitype_df.columns])
display(taxitype_nulls)

# COMMAND ----------

# Check for duplicate records
print("DUPLICATE RECORDS CHECK:")
print("-" * 60)
print(f"Total Drivers: {drivers_df.count()}")
print(f"Unique Driver IDs: {drivers_df.select('DriverID').distinct().count()}")
print(f"Total Taxis: {taxi_df.count()}")
print(f"Unique Taxi IDs: {taxi_df.select('TaxiID').distinct().count()}")
print(f"Unique License Plates: {taxi_df.select('LicensePlate').distinct().count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Drivers Analysis

# COMMAND ----------

# Basic statistics for drivers
print("DRIVER STATISTICS:")
print("=" * 60)
drivers_df.select('Age', 'YearsExperience', 'Rating').describe().show()

# COMMAND ----------

# Gender distribution
gender_dist = drivers_df.groupBy('Gender').agg(
    count('*').alias('Count'),
    round(avg('Age'), 2).alias('Avg_Age'),
    round(avg('Rating'), 2).alias('Avg_Rating')
).orderBy('Gender')

print("GENDER DISTRIBUTION:")
display(gender_dist)

# COMMAND ----------

# Convert to Pandas for visualization
gender_pd = gender_dist.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gender count
axes[0].bar(gender_pd['Gender'], gender_pd['Count'], color=['#3498db', '#e74c3c'])
axes[0].set_title('Driver Count by Gender', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Average rating by gender
axes[1].bar(gender_pd['Gender'], gender_pd['Avg_Rating'], color=['#2ecc71', '#f39c12'])
axes[1].set_title('Average Rating by Gender', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Average Rating')
axes[1].set_ylim([0, 5])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Age distribution analysis
age_bins = [20, 30, 40, 50, 60, 70]
drivers_with_age_group = drivers_df.withColumn(
    'AgeGroup',
    when(col('Age') < 30, '20-29')
    .when((col('Age') >= 30) & (col('Age') < 40), '30-39')
    .when((col('Age') >= 40) & (col('Age') < 50), '40-49')
    .when((col('Age') >= 50) & (col('Age') < 60), '50-59')
    .otherwise('60+')
)

age_group_stats = drivers_with_age_group.groupBy('AgeGroup').agg(
    count('*').alias('Count'),
    round(avg('Rating'), 2).alias('Avg_Rating'),
    round(avg('YearsExperience'), 2).alias('Avg_Experience')
).orderBy('AgeGroup')

print("DRIVERS BY AGE GROUP:")
display(age_group_stats)

# COMMAND ----------

# Create age groups and calculate statistics
age_group_stats = drivers_df.withColumn(
    'AgeGroup',
    when(col('Age') < 25, '18-24')
    .when((col('Age') >= 25) & (col('Age') < 35), '25-34')
    .when((col('Age') >= 35) & (col('Age') < 45), '35-44')
    .when((col('Age') >= 45) & (col('Age') < 55), '45-54')
    .otherwise('55+')
).groupBy('AgeGroup').agg(
    count('*').alias('Count'),
    avg('YearsExperience').alias('Avg_Experience'),  # Fixed: YearsExperience
    avg('Rating').alias('Avg_Rating')
).orderBy('AgeGroup')

# Show the statistics
age_group_stats.show()

# Visualize age distribution
age_pd = age_group_stats.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age group distribution
axes[0].bar(age_pd['AgeGroup'], age_pd['Count'], color='#3498db')
axes[0].set_title('Driver Count by Age Group', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Experience vs Rating by age group
axes[1].scatter(age_pd['Avg_Experience'], age_pd['Avg_Rating'], s=age_pd['Count']*20, alpha=0.6, color='#e74c3c')
for i, txt in enumerate(age_pd['AgeGroup']):
    axes[1].annotate(txt, (age_pd['Avg_Experience'].iloc[i], age_pd['Avg_Rating'].iloc[i]))
axes[1].set_title('Experience vs Rating by Age Group', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Years of Experience')
axes[1].set_ylabel('Average Rating')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Rating distribution
rating_dist = drivers_df.groupBy('Rating').count().orderBy('Rating')
print("RATING DISTRIBUTION:")
display(rating_dist)

# COMMAND ----------

# Experience analysis
experience_categories = drivers_df.withColumn(
    'ExperienceLevel',
    when(col('YearsExperience') < 3, 'Novice (0-2 years)')
    .when((col('YearsExperience') >= 3) & (col('YearsExperience') < 8), 'Intermediate (3-7 years)')
    .when((col('YearsExperience') >= 8) & (col('YearsExperience') < 15), 'Experienced (8-14 years)')
    .otherwise('Veteran (15+ years)')
)

exp_stats = experience_categories.groupBy('ExperienceLevel').agg(
    count('*').alias('Count'),
    round(avg('Rating'), 2).alias('Avg_Rating'),
    round(avg('Age'), 2).alias('Avg_Age')
).orderBy(
    when(col('ExperienceLevel') == 'Novice (0-2 years)', 1)
    .when(col('ExperienceLevel') == 'Intermediate (3-7 years)', 2)
    .when(col('ExperienceLevel') == 'Experienced (8-14 years)', 3)
    .otherwise(4)
)

print("DRIVERS BY EXPERIENCE LEVEL:")
display(exp_stats)

# COMMAND ----------

# Join date analysis
drivers_with_year = drivers_df.withColumn('JoinYear', year(col('JoinDate')))

join_year_stats = drivers_with_year.groupBy('JoinYear').agg(
    count('*').alias('Drivers_Joined'),
    round(avg('Rating'), 2).alias('Avg_Rating')
).orderBy('JoinYear')

print("DRIVER RECRUITMENT BY YEAR:")
display(join_year_stats)

# COMMAND ----------

# Visualize recruitment trends
join_pd = join_year_stats.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Drivers joined by year
axes[0].plot(join_pd['JoinYear'], join_pd['Drivers_Joined'], marker='o', linewidth=2, markersize=8, color='#2ecc71')
axes[0].fill_between(join_pd['JoinYear'], join_pd['Drivers_Joined'], alpha=0.3, color='#2ecc71')
axes[0].set_title('Driver Recruitment Trend', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Drivers Joined')
axes[0].grid(alpha=0.3)

# Average rating by join year
axes[1].bar(join_pd['JoinYear'], join_pd['Avg_Rating'], color='#9b59b6')
axes[1].set_title('Average Rating by Join Year', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Rating')
axes[1].set_ylim([0, 5])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Top performing drivers
top_drivers = drivers_df.orderBy(col('Rating').desc()).limit(10)
print("TOP 10 DRIVERS BY RATING:")
display(top_drivers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Taxi Fleet Analysis

# COMMAND ----------

# Taxi status distribution
status_dist = taxi_df.groupBy('Status').count().orderBy('Status')
print("TAXI STATUS DISTRIBUTION:")
display(status_dist)

# COMMAND ----------

# Visualize taxi status
status_pd = status_dist.toPandas()

fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)

ax.pie(status_pd['count'], labels=status_pd['Status'], autopct='%1.1f%%', 
       colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12})
ax.set_title('Taxi Fleet Status Distribution', fontsize=14, fontweight='bold', pad=20)
plt.show()

# COMMAND ----------

# Taxi type distribution
taxi_by_type = taxi_df.groupBy('TaxiTypeID').count().orderBy('TaxiTypeID')
print("TAXI DISTRIBUTION BY TYPE:")
display(taxi_by_type)

# COMMAND ----------

# Join with taxi type names for better understanding
taxi_with_type = taxi_df.join(taxitype_df, 'TaxiTypeID', 'left')

type_distribution = taxi_with_type.groupBy('TypeName').agg(
    count('*').alias('Count'),
    round(avg('BaseFare'), 2).alias('Avg_Base_Fare'),
    round(avg('PerKmRate'), 2).alias('Avg_PerKm_Rate')
).orderBy('Count', ascending=False)

print("TAXI FLEET BY TYPE NAME:")
display(type_distribution)

# COMMAND ----------

# Visualize taxi types
type_pd = type_distribution.toPandas()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Count by type
axes[0].barh(type_pd['TypeName'], type_pd['Count'], color='#3498db')
axes[0].set_title('Number of Taxis by Type', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].grid(axis='x', alpha=0.3)

# Pricing by type
x = range(len(type_pd))
width = 0.35
axes[1].bar([i - width/2 for i in x], type_pd['Avg_Base_Fare'], width, label='Base Fare', color='#2ecc71')
axes[1].bar([i + width/2 for i in x], type_pd['Avg_PerKm_Rate'], width, label='Per Km Rate', color='#e74c3c')
axes[1].set_title('Pricing Structure by Taxi Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Taxi Type')
axes[1].set_ylabel('Price ($)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(type_pd['TypeName'], rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Model distribution
model_dist = taxi_df.groupBy('Model').agg(
    count('*').alias('Count')
).orderBy('Count', ascending=False)

print("TAXI MODELS IN FLEET:")
display(model_dist)

# COMMAND ----------

# Year of manufacture analysis
year_stats = taxi_df.groupBy('YearManufactured').agg(
    count('*').alias('Count')
).orderBy('YearManufactured')

print("FLEET AGE DISTRIBUTION:")
display(year_stats)

# COMMAND ----------

# Calculate vehicle age
current_year = 2024
taxi_with_age = taxi_df.withColumn('VehicleAge', lit(current_year) - col('YearManufactured'))

age_group_dist = taxi_with_age.withColumn(
    'AgeCategory',
    when(col('VehicleAge') <= 2, 'New (0-2 years)')
    .when((col('VehicleAge') > 2) & (col('VehicleAge') <= 5), 'Recent (3-5 years)')
    .when((col('VehicleAge') > 5) & (col('VehicleAge') <= 8), 'Moderate (6-8 years)')
    .otherwise('Old (9+ years)')
).groupBy('AgeCategory').count().orderBy(
    when(col('AgeCategory') == 'New (0-2 years)', 1)
    .when(col('AgeCategory') == 'Recent (3-5 years)', 2)
    .when(col('AgeCategory') == 'Moderate (6-8 years)', 3)
    .otherwise(4)
)

print("FLEET AGE CATEGORIES:")
display(age_group_dist)

# COMMAND ----------

# Visualize fleet age
age_cat_pd = age_group_dist.toPandas()

plt.figure(figsize=(10, 6))
plt.bar(age_cat_pd['AgeCategory'], age_cat_pd['count'], color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
plt.title('Fleet Distribution by Vehicle Age', fontsize=14, fontweight='bold')
plt.xlabel('Age Category')
plt.ylabel('Number of Vehicles')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Comprehensive Fleet Overview

# COMMAND ----------

# Create a comprehensive fleet summary
fleet_summary = taxi_with_type.groupBy('TypeName', 'Status').agg(
    count('*').alias('Count')
).orderBy('TypeName', 'Status')

print("FLEET SUMMARY BY TYPE AND STATUS:")
display(fleet_summary)

# COMMAND ----------

# Pivot the data for better visualization
fleet_pivot = fleet_summary.groupBy('TypeName').pivot('Status').sum('Count').fillna(0)
display(fleet_pivot)

# COMMAND ----------

# Revenue potential analysis (assuming 10km average trip)
avg_trip_distance = 10

revenue_analysis = taxitype_df.withColumn(
    'Avg_Revenue_Per_Trip',
    round(col('BaseFare') + (col('PerKmRate') * lit(avg_trip_distance)), 2)
)

# Join with taxi counts
taxi_counts = taxi_df.groupBy('TaxiTypeID').agg(
    count('*').alias('Fleet_Count')
)

revenue_with_fleet = revenue_analysis.join(taxi_counts, 'TaxiTypeID', 'left').fillna(0)
revenue_with_fleet = revenue_with_fleet.withColumn(
    'Daily_Revenue_Potential',
    round(col('Avg_Revenue_Per_Trip') * col('Fleet_Count') * lit(10), 2)  # Assuming 10 trips per day
)

print("REVENUE POTENTIAL ANALYSIS (10km avg trip, 10 trips/day):")
display(revenue_with_fleet.select('TypeName', 'BaseFare', 'PerKmRate', 'Capacity', 
                                   'Fleet_Count', 'Avg_Revenue_Per_Trip', 'Daily_Revenue_Potential')
        .orderBy('Daily_Revenue_Potential', ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cross-Dataset Analysis

# COMMAND ----------

# Calculate total fleet utilization
total_taxis = taxi_df.count()
active_taxis = taxi_df.filter(col('Status') == 'Active').count()
maintenance_taxis = taxi_df.filter(col('Status') == 'Maintenance').count()

print("FLEET UTILIZATION METRICS:")
print("=" * 60)
print(f"Total Fleet Size: {total_taxis}")
print(f"Active Taxis: {active_taxis} ({__builtins__.round(active_taxis/total_taxis*100, 2)}%)")
print(f"In Maintenance: {maintenance_taxis} ({__builtins__.round(maintenance_taxis/total_taxis*100, 2)}%)")

# COMMAND ----------

# Driver to taxi ratio analysis
total_drivers = drivers_df.count()
driver_taxi_ratio = total_drivers / total_taxis

print("\nDRIVER-TAXI RATIO:")
print("=" * 60)
print(f"Total Drivers: {total_drivers}")
print(f"Total Taxis: {total_taxis}")
print(f"Ratio: {driver_taxi_ratio:.2f} drivers per taxi")

if driver_taxi_ratio > 1:
    surplus_pct = (driver_taxi_ratio - 1) * 100
    print(f"✓ Good coverage - {surplus_pct:.1f}% surplus drivers")
elif driver_taxi_ratio < 1:
    shortage = (1 - driver_taxi_ratio) * total_taxis
    print(f"⚠ Driver shortage - Need {shortage:.0f} more drivers")

# COMMAND ----------

# Capacity analysis
capacity_summary = taxitype_df.join(
    taxi_df.groupBy('TaxiTypeID').count(), 'TaxiTypeID', 'left'
).fillna(0)

capacity_summary = capacity_summary.withColumn(
    'Total_Capacity',
    col('Capacity') * col('count')
).select('TypeName', 'Capacity', 'count', 'Total_Capacity').orderBy('Total_Capacity', ascending=False)

print("FLEET CAPACITY ANALYSIS:")
display(capacity_summary)

total_capacity = capacity_summary.agg(sum('Total_Capacity')).collect()[0][0]
print(f"\nTotal Fleet Passenger Capacity: {total_capacity} passengers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key Insights and Recommendations

# COMMAND ----------

# Generate insights
print("=" * 80)
print("KEY INSIGHTS AND RECOMMENDATIONS")
print("=" * 80)

# 1. Driver insights
avg_rating = drivers_df.agg(avg('Rating')).collect()[0][0]
avg_experience = drivers_df.agg(avg('YearsExperience')).collect()[0][0]
high_rated = drivers_df.filter(col('Rating') >= 4.5).count()
high_rated_pct = high_rated / total_drivers * 100

print("\n1. DRIVER INSIGHTS:")
print(f"   • Average driver rating: {avg_rating:.2f}/5.0")
print(f"   • Average experience: {avg_experience:.2f} years")
print(f"   • High-rated drivers (≥4.5): {high_rated} ({high_rated_pct:.1f}%)")

# 2. Fleet insights
most_common_type = type_distribution.orderBy('Count', ascending=False).first()['TypeName']
oldest_vehicle = taxi_df.agg(min('YearManufactured')).collect()[0][0]
newest_vehicle = taxi_df.agg(max('YearManufactured')).collect()[0][0]
fleet_utilization_pct = active_taxis / total_taxis * 100

print("\n2. FLEET INSIGHTS:")
print(f"   • Most common taxi type: {most_common_type}")
print(f"   • Fleet age range: {oldest_vehicle} to {newest_vehicle}")
print(f"   • Fleet utilization: {fleet_utilization_pct:.1f}% active")

# 3. Revenue insights
best_revenue_type = revenue_with_fleet.orderBy('Daily_Revenue_Potential', ascending=False).first()

print("\n3. REVENUE POTENTIAL:")
print(f"   • Highest revenue type: {best_revenue_type['TypeName']}")
print(f"   • Daily potential: ${best_revenue_type['Daily_Revenue_Potential']:,.2f}")

# 4. Recommendations
print("\n4. RECOMMENDATIONS:")
print("   • Consider recruiting more drivers to improve coverage")
print("   • Monitor vehicles in maintenance to minimize downtime")
print("   • Focus on high-capacity vehicles for revenue optimization")
print("   • Implement driver training programs to maintain high ratings")
print("   • Plan for fleet renewal of vehicles older than 8 years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Export Summary Statistics

# COMMAND ----------

# Create a summary report with all numeric values as floats
summary_stats = spark.createDataFrame([
    ("Total Drivers", float(total_drivers)),
    ("Total Taxis", float(total_taxis)),
    ("Active Taxis", float(active_taxis)),
    ("Taxis in Maintenance", float(maintenance_taxis)),
    ("Average Driver Rating", float(f"{avg_rating:.2f}")),
    ("Average Driver Experience (years)", float(f"{avg_experience:.2f}")),
    ("Total Fleet Capacity", float(total_capacity)),
    ("Driver-Taxi Ratio", float(f"{driver_taxi_ratio:.2f}"))
], ["Metric", "Value"])

print("SUMMARY STATISTICS:")
display(summary_stats)

# COMMAND ----------

# Save summary statistics to a temporary view
summary_stats.createOrReplaceTempView("fleet_summary_stats")
print("✓ Summary statistics saved to temporary view: fleet_summary_stats")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Next Steps
# MAGIC
# MAGIC This notebook has provided a comprehensive exploratory data analysis of the taxi fleet. Here are suggested next steps:
# MAGIC
# MAGIC 1. **Predictive Analytics**: Build models to predict driver performance or taxi maintenance needs
# MAGIC 2. **Route Optimization**: Analyze trip data to optimize driver routes
# MAGIC 3. **Demand Forecasting**: Predict peak demand times and locations
# MAGIC 4. **Customer Segmentation**: Analyze customer preferences by taxi type
# MAGIC 5. **Real-time Dashboards**: Create live monitoring dashboards for fleet operations
# MAGIC 6. **Cost Analysis**: Detailed analysis of operational costs vs revenue
# MAGIC 7. **Driver Performance**: Deep dive into factors affecting driver ratings
# MAGIC 8. **Fleet Optimization**: Determine optimal fleet composition

# COMMAND ----------

# DBTITLE 1,Visualize age distribution
# Visualize age distribution
age_pd = age_group_stats.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age group distribution
axes[0].bar(age_pd['AgeGroup'], age_pd['Count'], color='#3498db')
axes[0].set_title('Driver Count by Age Group', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(age_pd['AgeGroup'], rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Experience vs Rating by age group
axes[1].scatter(age_pd['Avg_Experience'], age_pd['Avg_Rating'], s=age_pd['Count']*20, alpha=0.6, color='#e74c3c')
for i, txt in enumerate(age_pd['AgeGroup']):
    axes[1].annotate(txt, (age_pd['Avg_Experience'].iloc[i], age_pd['Avg_Rating'].iloc[i]))
axes[1].set_title('Experience vs Rating by Age Group', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Years of Experience')
axes[1].set_ylabel('Average Rating')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
