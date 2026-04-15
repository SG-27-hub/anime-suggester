from flask import Flask, request, render_template, jsonify, redirect
import json
from sentence_transformers import SentenceTransformer, util
import random
import os
from google import genai 
from functools import lru_cache 

app = Flask(__name__)

# --- Configure Gemini API ---
my_api_key = os.environ.get("GEMINI_API_KEY")
if my_api_key:
    client = genai.Client()
else:
    print("Warning: No Gemini API Key found in environment variables.")
    client = None

# 1. Load the AI Model
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load database and clean inconsistent data
with open("anime.json", "r", encoding="utf-8") as file:
    anime_list = json.load(file)

for anime in anime_list:
    # Fix inconsistent spacing in ratings (e.g., "8.1 / 10" -> "8.1/10")
    rating_str = str(anime.get("rating", "0/10")).replace(" ", "")
    anime["rating"] = rating_str
    
    # Create a numeric rating for dynamic sorting
    try:
        anime['numeric_rating'] = float(rating_str.split('/')[0])
    except ValueError:
        anime['numeric_rating'] = 0.0

# 3. Pre-calculate embeddings for all descriptions
print("Generating database vectors...")
descriptions = [a["desc"] for a in anime_list]
corpus_embeddings = model.encode(descriptions, convert_to_tensor=True)

# --- Cached Gemini Function ---
@lru_cache(maxsize=100)
def get_gemini_response(user_prompt, title, desc, history_text=""):
    """Caches identical prompts to save API quota."""
    if not client:
        return f"I found a great match for you: **{title}**."
        
    system_prompt = f"""
    You are a friendly AI romance anime recommender.
    The user asked for: "{user_prompt}"
    Recent conversation history:
    {history_text}
    
    Based on their request and history, you have selected the anime: "{title}".
    Here is the synopsis of the anime: "{desc}"
    
    Write a short, engaging 2-3 sentence response telling the user exactly why this specific anime fits what they asked for. 
    Focus on elements that suggest a happy ending or satisfying conclusion. Ensure the recommendation focuses on standard romance (exclude BL/GL genres).
    Use bullet points if you need to list anything. Do not use any emojis whatsoever.
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview', 
            contents=system_prompt
        )
        return response.text
    except Exception as e:
        print("Gemini API Error:", e, flush=True) 
        # GRACEFUL FALLBACK MESSAGE
        return f"I found a great match for you: **{title}**. It fits your mood perfectly and has a satisfying conclusion."

@app.route("/")
def home():
    titles_json = json.dumps([a["title"] for a in anime_list])
    skip_splash = request.args.get("skip_splash")
    
    # --- FIXED: Trending Anime Logic ---
    # First try to find anime explicitly marked as trending
    trending_anime = [a for a in anime_list if a.get("trending", False)]
    
    # If the trending field doesn't exist or is empty, dynamically grab the top 4 highest rated
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
        trending_anime=trending_anime 
    )

@app.route("/suggest")
def suggest():
    mood = request.args.get("mood", "all")
    search_query = request.args.get("search", "")
    
    # 1. If the search bar is empty, pick a random anime
    if not search_query:
        if mood != "all":
            filtered_list = [a for a in anime_list if mood.lower() in [g.lower() for g in a.get("genre", [])]]
            if filtered_list:
                anime = random.choice(filtered_list)
            else:
                anime = random.choice(anime_list)
        else:
            anime = random.choice(anime_list)
            
        return render_template("recommendation.html", anime=anime, mood=mood)

    # 2. Check for a direct Title match FIRST
    search_lower = search_query.lower()
    for a in anime_list:
        if search_lower in a["title"].lower():
            return render_template("recommendation.html", anime=a, mood=mood)

    # 3. AI Semantic Search
    query_embedding = model.encode(search_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
    
    best_match_idx = hits[0][0]['corpus_id']
    anime = anime_list[best_match_idx]

    return render_template("recommendation.html", anime=anime, mood=mood)

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    history = data.get("history", [])

    if not user_prompt:
        return jsonify({"error": "Empty prompt"})
        
    # Format the history into a single string for the AI prompt
    # Only keeping the last few messages to prevent token bloat
    formatted_history = []
    for msg in history[-4:]:
        role = "User" if msg.get("role") == "user" else "AI"
        text = msg.get("text", "")
        formatted_history.append(f"{role}: {text}")
    history_text = "\n".join(formatted_history)

    found_anime = None

    # 1. Check for a direct Title match FIRST
    prompt_lower = user_prompt.lower()
    for a in anime_list:
        title_lower = a["title"].lower()
        clean_title = title_lower.replace("!", "").replace("?", "") 
        
        if prompt_lower in title_lower or clean_title in prompt_lower:
            found_anime = a
            break 

    # 2. AI Semantic Search 
    if not found_anime:
        query_embedding = model.encode(user_prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
        
        chosen_hit = random.choice(hits[0])
        best_match_idx = chosen_hit['corpus_id']
        found_anime = anime_list[best_match_idx]

    # 3. Generate response using the cached function with history included
    ai_text = get_gemini_response(user_prompt, found_anime['title'], found_anime['desc'], history_text)
    
    found_anime['ai_message'] = ai_text

    return jsonify(found_anime)

# --- Rating Endpoint ---
@app.route("/api/rate", methods=["POST"])
def rate_anime():
    data = request.get_json()
    title = data.get("title")
    rating = data.get("rating")
    
    if not title or not rating:
         return jsonify({"error": "Missing title or rating"}), 400
         
    rating_entry = {"title": title, "rating": rating}
    ratings_file = "ratings.json"
    
    # Load existing ratings or create a new list if it doesn't exist
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
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
