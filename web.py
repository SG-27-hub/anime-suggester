from flask import Flask, request, render_template, jsonify, redirect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
from sentence_transformers import SentenceTransformer, util
import random
import os
from google import genai 
import traceback
from werkzeug.exceptions import HTTPException
import torch

app = Flask(__name__)

# Initialize the rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://"
)

# --- Configure Gemini API ---
my_api_key = os.environ.get("GEMINI_API_KEY")
if my_api_key:
    client = genai.Client()
else:
    print("Warning: No Gemini API Key found in environment variables.")
    client = None

# --- Load and Clean Database ---
with open("anime.json", "r", encoding="utf-8") as file:
    raw_anime_list = json.load(file)

anime_list = []

# Define strict negative keywords to filter out unwanted niches
negative_filters = ["yaoi", "shounen ai", "boys love", "tragedy", "sad ending", "bittersweet"]

for anime in raw_anime_list:
    # Combine the genre list and description into one lowercase string for easy checking
    content_check = (str(anime.get("genre", [])) + " " + anime.get("desc", "")).lower()
    
    # If any negative keyword is found, skip adding this anime to the site entirely
    if any(keyword in content_check for keyword in negative_filters):
        continue
        
    # Fix inconsistent spacing in ratings
    rating_str = str(anime.get("rating", "0/10")).replace(" ", "")
    anime["rating"] = rating_str
    
    try:
        anime['numeric_rating'] = float(rating_str.split('/')[0])
    except ValueError:
        anime['numeric_rating'] = 0.0
        
    anime_list.append(anime)

# --- Lazy Load AI Model & Vectors ---
model = None
corpus_embeddings = None

def get_model_and_embeddings():
    global model, corpus_embeddings
    
    if model is None:
        print("Lazy Loading AI Model...", flush=True)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
    if corpus_embeddings is None:
        embeddings_file = "embeddings.pt"
        if os.path.exists(embeddings_file):
            print("Loading saved database vectors from disk...", flush=True)
            corpus_embeddings = torch.load(embeddings_file)
        else:
            print("Generating database vectors...", flush=True)
            descriptions = [a["desc"] for a in anime_list]
            corpus_embeddings = model.encode(descriptions, convert_to_tensor=True)
            torch.save(corpus_embeddings, embeddings_file)
            
    return model, corpus_embeddings

# --- Routes ---
@app.route("/")
def home():
    titles_json = json.dumps([a["title"] for a in anime_list])
    skip_splash = request.args.get("skip_splash")
    
    # Calculate how many anime belong to each mood
    mood_counts = {
        "all": len(anime_list),
        "school": len([a for a in anime_list if "school" in [g.lower() for g in a.get("genre", [])]]),
        "comedy": len([a for a in anime_list if "comedy" in [g.lower() for g in a.get("genre", [])]]),
        "fantasy": len([a for a in anime_list if "fantasy" in [g.lower() for g in a.get("genre", [])]])
    }
    
    trending_anime = [a for a in anime_list if a.get("trending", False)]
    if not trending_anime:
        sorted_by_rating = sorted(anime_list, key=lambda x: x.get('numeric_rating', 0), reverse=True)
        trending_anime = sorted_by_rating[:4]
    else:
        trending_anime = trending_anime[:4]
    
    return render_template(
        "index.html",
        titles_json=titles_json,
        anime_count=len(anime_list),
        skip_splash=skip_splash,
        trending_anime=trending_anime,
        mood_counts=mood_counts
    )

@app.route("/suggest")
def suggest():
    mood = request.args.get("mood", "all")
    search_query = request.args.get("search", "")
    
    if not search_query:
        if mood != "all":
            filtered_list = [a for a in anime_list if mood.lower() in [g.lower() for g in a.get("genre", [])]]
            anime = random.choice(filtered_list) if filtered_list else random.choice(anime_list)
        else:
            anime = random.choice(anime_list)
            
        return render_template("recommendation.html", anime=anime, mood=mood)

    search_lower = search_query.lower()
    for a in anime_list:
        if search_lower in a["title"].lower():
            return render_template("recommendation.html", anime=a, mood=mood)

    local_model, local_embeddings = get_model_and_embeddings()
    query_embedding = local_model.encode(search_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, local_embeddings, top_k=1)
    
    best_match_idx = hits[0][0]['corpus_id']
    anime = anime_list[best_match_idx]

    return render_template("recommendation.html", anime=anime, mood=mood)

@app.route("/anime/<path:title>")
def anime_detail(title):
    title_lower = title.lower()
    for a in anime_list:
        if a["title"].lower() == title_lower:
            return render_template("recommendation.html", anime=a, mood="all")
            
    return redirect("/?skip_splash=1")

@app.route("/api/similar")
def get_similar():
    title = request.args.get("title")
    if not title:
        return jsonify([])

    idx = next((i for i, a in enumerate(anime_list) if a["title"] == title), None)
    if idx is None:
        return jsonify([])

    local_model, local_embeddings = get_model_and_embeddings()
    cos_scores = util.cos_sim(local_embeddings[idx], local_embeddings)[0]
    
    top_results = torch.topk(cos_scores, k=5)
    similar_indices = top_results.indices.tolist()
    
    results = []
    for i in similar_indices:
        if anime_list[i]["title"] != title:
            results.append({
                "title": anime_list[i]["title"],
                "img": anime_list[i].get("img", "")
            })
            
    return jsonify(results[:4])
    
@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/api/ask", methods=["POST"])
@limiter.limit("5 per minute")
def api_ask():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    history = data.get("history", [])

    if not user_prompt:
        return jsonify({"error": "Empty prompt"})

    clean_prompt = user_prompt.lower().strip()
    greetings = ["hi", "hello", "hey", "yo", "good morning", "good evening", "sup"]
    vague_requests = ["suggest me an anime", "recommend an anime", "give me an anime", "what should i watch"]

    if clean_prompt in greetings:
        return jsonify({
            "ai_message": "Hello! I am your Romance Anime AI. How can I help you today? You can tell me what kind of mood, setting, or tropes you are looking for!",
            "title": None
        })
    
    if clean_prompt in vague_requests:
        return jsonify({
            "ai_message": "I would love to give you a recommendation! To find the perfect match, tell me a bit more about what you want. Do you prefer comedy, fantasy, school life, or maybe a slow-burn romance?",
            "title": None
        })

    found_anime = None

    prompt_lower = user_prompt.lower()
    for a in anime_list:
        title_lower = a["title"].lower()
        clean_title = title_lower.replace("!", "").replace("?", "") 
        if prompt_lower in title_lower or clean_title in prompt_lower:
            found_anime = a
            break 

    if not found_anime:
        local_model, local_embeddings = get_model_and_embeddings()
        query_embedding = local_model.encode(user_prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, local_embeddings, top_k=1)
        best_match_idx = hits[0][0]['corpus_id']
        found_anime = anime_list[best_match_idx]

    system_prompt = f"""
    ROLE: Friendly AI Romance Expert.
    STRICT RULE 1: DO NOT use any emojis.
    STRICT RULE 2: Use bullet points for lists.
    STRICT RULE 3: Suggest standard romance only.
    
    CONTEXT: The user is currently recommended to view "{found_anime['title']}".
    Synopsis: "{found_anime['desc']}"
    Explain in 2-3 sentences why this anime is a perfect match based on their request.
    """

    # --- BUG FIX: Intercept and filter the history array ---
    valid_history = history.copy()
    if valid_history and valid_history[-1].get("role") == "user" and valid_history[-1].get("content") == user_prompt:
        valid_history.pop()

    # Format the frontend history for the genai SDK
    formatted_history = []
    for msg in valid_history[-4:]:
        role = "user" if msg.get("role") == "user" else "model"
        formatted_history.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })

    if client:
        try:
            chat = client.chats.create(
                model='gemini-3.1-flash-lite-preview',
                config={"system_instruction": system_prompt},
                history=formatted_history
            )
            response = chat.send_message(user_prompt)
            ai_text = response.text
        except Exception as e:
            print("Gemini API Error:", e, flush=True) 
            ai_text = f"I am experiencing heavy traffic right now, but I still highly recommend **{found_anime['title']}**. It fits exactly what you are looking for and guarantees a great happy ending!"
    else:
        ai_text = f"I found a great match for you: **{found_anime['title']}**."
        
    found_anime['ai_message'] = ai_text
    return jsonify(found_anime)

@app.route("/api/rate", methods=["POST"])
def rate_anime():
    data = request.get_json()
    title = data.get("title")
    rating = data.get("rating")
    
    if not title or not rating:
         return jsonify({"error": "Missing title or rating"}), 400
         
    rating_entry = {"title": title, "rating": rating}
    ratings_file = "ratings.json"
    
    if os.path.exists(ratings_file):
        try:
            with open(ratings_file, "r", encoding="utf-8") as f:
                ratings = json.load(f)
        except json.JSONDecodeError:
            ratings = []
    else:
        ratings = []
        
    ratings.append(rating_entry)
    
    with open(ratings_file, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=4)
        
    return jsonify({"success": True, "message": "Rating saved!"})

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    
    print(f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", flush=True)
    return "<p>An unexpected error occurred. Please try again later.</p>", 500
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
