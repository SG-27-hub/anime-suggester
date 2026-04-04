from flask import Flask, request, render_template, jsonify, redirect
import json
from sentence_transformers import SentenceTransformer, util
import random
import os
from google import genai # NEW IMPORT

app = Flask(__name__)

# --- NEW: Configure Gemini API ---
my_api_key = os.environ.get("GEMINI_API_KEY")
if my_api_key:
    # The new client automatically finds your key in the Hugging Face secrets!
    client = genai.Client()
else:
    print("Warning: No Gemini API Key found in environment variables.")
    client = None

# 1. Load the AI Model
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ... (keep loading your JSON database as normal) ...
with open("anime.json", "r", encoding="utf-8") as file:
    anime_list = json.load(file)

# 3. Pre-calculate embeddings for all descriptions
print("Generating database vectors...")
descriptions = [a["desc"] for a in anime_list]
corpus_embeddings = model.encode(descriptions, convert_to_tensor=True)

@app.route("/")
def home():
    titles_json = json.dumps([a["title"] for a in anime_list])
    skip_splash = request.args.get("skip_splash")
    
    return render_template(
        "index.html",
        titles_json=titles_json,
        anime_count=len(anime_list),
        skip_splash=skip_splash
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
            # If the search matches a title, return this anime immediately
            return render_template("recommendation.html", anime=a, mood=mood)

    # 3. AI Semantic Search (Only runs if the search didn't match a title)
    query_embedding = model.encode(search_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
    
    best_match_idx = hits[0][0]['corpus_id']
    anime = anime_list[best_match_idx]

    return render_template("recommendation.html", anime=anime, mood=mood)

@app.route("/chat")
def chat_page():
    # This just loads the visual chat interface
    return render_template("chat.html")

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"error": "Empty prompt"})

    found_anime = None

    # 1. Check for a direct Title match FIRST
    prompt_lower = user_prompt.lower()
    for a in anime_list:
        title_lower = a["title"].lower()
        clean_title = title_lower.replace("!", "").replace("?", "") 
        
        if prompt_lower in title_lower or clean_title in prompt_lower:
            found_anime = a
            break # Stop the loop, we found a match

    # 2. AI Semantic Search (Only runs if no title was mentioned)
    if not found_anime:
        query_embedding = model.encode(user_prompt, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
        
        chosen_hit = random.choice(hits[0])
        best_match_idx = chosen_hit['corpus_id']
        found_anime = anime_list[best_match_idx]

    # 3. Generate a real AI response using Gemini
    if client:
        # We build a private instruction telling Gemini exactly what to say
        system_prompt = f"""
        You are a friendly AI romance anime recommender.
        The user asked for: "{user_prompt}"
        Based on their request, you have selected the anime: "{found_anime['title']}".
        Here is the synopsis of the anime: "{found_anime['desc']}"
        
        Write a short, engaging 2-3 sentence response telling the user exactly why this specific anime fits what they asked for. Use bullet points if you need to list anything, but do not use any emojis.
        """
        
        try:
            # Ask Gemini to generate the text using the NEW package
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=system_prompt
            )
            ai_text = response.text
        except Exception as e:
            # Added flush=True to force the log to print immediately
            print("Gemini API Error:", e, flush=True) 
            
            # Temporarily send the exact error directly to the chat window!
            ai_text = f"- ERROR: {str(e)}"
    else:
        # Fallback if the API key isn't working
        ai_text = f"I found a great match for you: **{found_anime['title']}**."
    
    # Add the generated text to the anime data being sent back to chat.html
    found_anime['ai_message'] = ai_text

    return jsonify(found_anime)
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
