from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# MongoDB connection string (Replace with your credentials if needed)
uri = "YOUR_API_KEY"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Select the database and collection
db = client['my_database']  # Replace 'my_database' with your database name
collection = db['results']  # Replace 'results' with your collection name

# Sample data to insert into the collection
sample_data = [
    {
        "pdf_name": "Sample_PDF_1",
        "summary": "This is the summary of Sample_PDF_1. It contains information about AI and its applications.",
        "keywords": ["AI", "applications", "machine learning", "deep learning"],
        "timestamp": "2024-10-11 12:30"
    },
    {
        "pdf_name": "Sample_PDF_2",
        "summary": "This is the summary of Sample_PDF_2. It discusses the impacts of climate change on the environment.",
        "keywords": ["climate change", "environment", "global warming", "sustainability"],
        "timestamp": "2024-10-11 12:45"
    }
]

# Insert the sample data into the collection
try:
    result = collection.insert_many(sample_data)
    print(f"Inserted {len(result.inserted_ids)} documents into the collection.")
except Exception as e:
    print(f"Error inserting data: {e}")
