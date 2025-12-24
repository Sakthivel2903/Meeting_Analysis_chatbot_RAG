>Create a .env file on same folder 
>Enter the api key of Groq 
>Enter the connection string of mongo db atlas

IMPORTANT:
Step-by-Step: MongoDB Atlas Vector Search Setup
Step 1: Go to MongoDB Atlas Dashboard

Open your browser and go to: https://cloud.mongodb.com/
Log in to your MongoDB Atlas account
Select your Cluster (the one you're using for this project)


Step 2: Navigate to Atlas Search

In your cluster view, click on the "Atlas Search" tab (or "Search" tab)
You'll see a button that says "Create Search Index"
Click "Create Search Index"


Step 3: Choose JSON Editor

You'll see two options:

Visual Editor
JSON Editor


Select "JSON Editor" (click "Next")


Step 4: Select Database and Collection

Database: Select meeting_db
Collection: Select document_chunks
Index Name: Enter vector_index (this must match the name in the code)


Step 5: Paste the JSON Configuration
In the JSON editor, paste this exact configuration:
json{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}


