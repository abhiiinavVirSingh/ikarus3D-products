import os
import io
import joblib
import pandas as pd
import numpy as np
import ast 
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional


from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import requests
import torch # Import torch

# Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv

# LangChain
from langchain_core.prompts import PromptTemplate
# Use the updated import path if using newer langchain versions
# from langchain_huggingface import HuggingFacePipeline
# New/Correct import
from langchain_huggingface import HuggingFacePipeline

# --- Configuration & Model Loading ---
load_dotenv(dotenv_path='PineCone_API.env') # Load environment variables from PineCone_API.env # Load environment variables from .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ikarus-products" # Make sure this matches your index in Pinecone
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [idx['name'] for idx in pc.list_indexes().indexes]:
     raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
index = pc.Index(PINECONE_INDEX_NAME)

# Load models (consider doing this once on startup)
print("Loading models...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # Load only if needed for classification endpoint
# clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # Load only if needed
# clf_model = joblib.load('backend/models/clip_linear_classifier.joblib') # Load only if needed
# label_space = joblib.load('backend/models/label_space.joblib') # Load only if needed

# Load data for lookups/analytics
try:
    df_products = pd.read_csv('backend/intern_data_ikarus.csv')
    # Basic cleaning might be needed again if IDs are not strings
    df_products['uniq_id'] = df_products['uniq_id'].astype(str)
    df_products.set_index('uniq_id', inplace=True) # Useful for quick lookups by ID
    print("Product data loaded.")
except FileNotFoundError:
    print("Warning: Product data CSV not found. Some features might not work.")
    df_products = None

# --- LangChain Setup ---
# Use a GPU if available, otherwise CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

gen_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=100, # Adjust as needed
    device=device
)
llm = HuggingFacePipeline(pipeline=gen_pipeline)

prompt_template = PromptTemplate.from_template(
    "Write a creative, vivid but concise product blurb (50-80 words) for a furniture item.\n"
    "Title: {title}\nBrand: {brand}\nMaterial: {material}\nColor: {color}\nCategories: {categories}\n"
    "Focus on the key benefits and style. Make it friendly and helpful for shoppers."
)

print("Models and data loaded successfully.")

# --- FastAPI App ---
app = FastAPI()

# --- CORS Middleware ---
# Allows requests from your React frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust if your React app runs elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---
class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5 # Number of recommendations to return

class ProductMetadata(BaseModel):
    uniq_id: str
    title: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    categories: Optional[List[str]] = None
    images: Optional[List[str]] = None
    material: Optional[str] = None
    color: Optional[str] = None
    price: Optional[float] = None
    cluster: Optional[int] = None

class RecommendationResponse(BaseModel):
    uniq_id: str
    score: float
    metadata: Optional[ProductMetadata] = None
    generated_blurb: Optional[str] = None

class AnalyticsData(BaseModel):
     # Define structure based on what analytics you want
     category_counts: Optional[dict] = None
     price_distribution: Optional[dict] = None # e.g., {'min': ..., 'max': ..., 'avg': ...}
     # Add more analytics fields as needed

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Ikarus Product Recommendation API"}

@app.post("/recommend", response_model=List[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    try:
        print(f"Received query: {request.query}")
        # 1. Embed the query
        query_embedding = embed_model.encode(request.query).tolist()

        # 2. Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )

        recommendations = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            # Ensure metadata values are correctly typed before creating Pydantic model
            metadata_typed = {
                "uniq_id": str(metadata.get("uniq_id", match.get("id", ""))), # Ensure ID is string
                "title": metadata.get("title"),
                "brand": metadata.get("brand"),
                "description": metadata.get("description"),
                "categories": metadata.get("categories", []),
                "images": metadata.get("images", []),
                "material": metadata.get("material"),
                "color": metadata.get("color"),
                "price": metadata.get("price"),
                "cluster": metadata.get("cluster")
            }
            product_meta = ProductMetadata(**metadata_typed)

            # 3. Generate Blurb using LangChain LLM
            try:
                formatted_prompt = prompt_template.format(
                    title=product_meta.title or "N/A",
                    brand=product_meta.brand or "N/A",
                    material=product_meta.material or "N/A",
                    color=product_meta.color or "N/A",
                    categories=' > '.join(product_meta.categories[:3]) if product_meta.categories else "N/A"
                )
                generated_blurb = llm.invoke(formatted_prompt) # Use invoke for newer langchain
                # generated_blurb = llm(formatted_prompt) # Use call for older langchain
            except Exception as llm_error:
                print(f"Error generating blurb for {product_meta.uniq_id}: {llm_error}")
                generated_blurb = "Could not generate description at this time."


            recommendations.append(RecommendationResponse(
                uniq_id=str(match['id']), # Ensure ID is string
                score=match['score'],
                metadata=product_meta,
                generated_blurb=generated_blurb.strip()
            ))

        print(f"Returning {len(recommendations)} recommendations.")
        return recommendations

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Analytics Endpoint (Example) ---
@app.get("/analytics", response_model=AnalyticsData)
async def get_analytics():
    if df_products is None:
         raise HTTPException(status_code=503, detail="Product data not available for analytics.")

    try:
        # Example: Category counts (using the parsed categories from your notebook)
        # Re-apply category parsing if not done during loading
        if 'categories_list' not in df_products.columns: # Check if parsing is needed
             def parse_list_safe(x):
                 if isinstance(x, str):
                     try: return ast.literal_eval(x)
                     except: return []
                 return x if isinstance(x, list) else []
             df_products['categories_list'] = df_products['categories'].apply(parse_list_safe)

        # Get top-level category
        df_products['top_category'] = df_products['categories_list'].apply(lambda x: x[0] if (isinstance(x, list) and len(x) > 0) else 'Unknown')
        category_counts = df_products['top_category'].value_counts().to_dict()

        # Example: Price distribution
        price_stats = df_products['price'].agg(['min', 'max', 'mean', 'median']).to_dict()
        price_distribution = {k: (v if pd.notna(v) else None) for k, v in price_stats.items()} # Handle NaN

        return AnalyticsData(
            category_counts=category_counts,
            price_distribution=price_distribution
        )
    except Exception as e:
        print(f"Error during analytics generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {e}")


# --- Image Classification Endpoint (Optional) ---
# Add if you want to expose the CV model directly
# @app.post("/classify")
# async def classify_image(image_url: HttpUrl):
#    try:
#        img = Image.open(io.BytesIO(requests.get(str(image_url), timeout=10).content)).convert("RGB")
#        inputs = clip_proc(images=img, return_tensors="pt")
#        with torch.no_grad():
#            image_features = clip_model.get_image_features(**inputs)
#        features_np = image_features[0].cpu().numpy().reshape(1, -1) # Reshape for classifier
#        prediction = clf_model.predict(features_np)
#        predicted_label = label_space[prediction[0]] # Map index back to label if needed
#        return {"category": predicted_label}
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=f"Error classifying image: {e}")


# To run: uvicorn main:app --reload